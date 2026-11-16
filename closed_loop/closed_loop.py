"""
Closed-loop simulation: do better error predictions give better circuits?

Run (after adapter_calibration.py):
    pip3 install qiskit-aer          # once
    python3 closed_loop.py           # all machines, default settings
    python3 closed_loop.py --machines ibm_fez --day-step 5     # quicker first look

For each test day (second half of the history, as in the backtests) and each queue delay D:
  1. PREDICT  error maps (2Q gate, 1Q gate, readout) with lcd, w10 and asym, using only data
              measured at least D hours before the job runs (job runs at 12:00 UTC that day).
              Links IBM had flagged broken in their latest report are removed for every method.
  2. COMPILE  each benchmark circuit with the SAME Qiskit compiler (optimization level 3: its
              VF2Layout / VF2PostLayout passes pick qubits using the error map) and the same seed.
              Only the error map differs between methods.
  3. TRUE DEVICE = the calibration actually measured right after the job time
              (next report of each gate / qubit; a link reported broken counts as error 0.5).
  4. SCORE    ESP  = product of (1 - true error) over every gate and measurement used (all days)
              fidelity = 1 - TVD vs the ideal output, from a noisy Aer simulation with the true
              calibration (2Q/1Q errors, readout, T1/T2), on every --full-sim-every-th day.
Outputs: results_cl/esp_<machine>.csv, results_cl/sim_<machine>.csv, printed summary,
         results_cl/gain.png
Caveats: the true device is the next calibration, not real hardware; no crosstalk, no idle
decoherence, no intra-day drift.
"""
import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ["lcd", "w10", "asym"]
STALE_DAYS = 7          # a prediction needs a measurement in the last 7 days
TRUTH_DAYS = 3          # the "true" value is the first report within 3 days after the job
BROKEN_ERR = 0.5        # true error of a link IBM reports as broken
EPOCH = pd.Timestamp("2020-01-01", tz="UTC")
OUT = Path("results_cl")


# ---------------------------------------------------------------- calibration series
def build_series(cal, h_down, h_up):
    """Per metric and element: report times, broken flags, good values, asym states."""
    S = {}
    for metric, gm in cal.groupby("metric"):
        gm = gm.sort_values(["element", "t"])
        gaps = gm.groupby("element").t.diff().dt.total_seconds() / 86400
        cycle = float(np.nanmedian(gaps)) if gaps.notna().any() else 1.0
        sub = {}
        for el, g in gm.groupby("element"):
            t_all = ((g.t - EPOCH).dt.total_seconds() / 86400).to_numpy()
            v_all = g.value.to_numpy()
            broken = v_all >= 1 if metric == "2q" else np.zeros(len(v_all), bool)
            t, v = t_all[~broken], v_all[~broken]
            if len(t) == 0:
                sub[el] = dict(t_all=t_all, v_all=v_all, broken=broken, t=t, v=v)
                continue
            y = np.log10(v)
            state = np.empty(len(y))
            m = y[0]
            state[0] = m
            for k in range(1, len(y)):
                dc = (t[k] - t[k - 1]) / cycle
                lam = 0.5 ** (dc / (h_down if y[k] < m else h_up))
                m = lam * m + (1 - lam) * y[k]
                state[k] = m
            sub[el] = dict(t_all=t_all, v_all=v_all, broken=broken, t=t, v=v, y=y, state=state,
                           cs=np.concatenate([[0.0], np.cumsum(y)]))
        S[metric] = sub
    return S


def predict(sub, t_cut, method, window):
    out = {}
    for el, s in sub.items():
        ia = np.searchsorted(s["t_all"], t_cut, side="right") - 1
        if ia < 0 or s["broken"][ia]:
            continue                                     # never measured, or flagged broken
        i = np.searchsorted(s["t"], t_cut, side="right") - 1
        if i < 0 or t_cut - s["t"][i] > STALE_DAYS:
            continue
        if method == "lcd":
            yh = s["y"][i]
        elif method == "w10":
            j0 = np.searchsorted(s["t"], s["t"][i] - window, side="left")
            yh = (s["cs"][i + 1] - s["cs"][j0]) / (i + 1 - j0)
        else:
            yh = s["state"][i]
        out[el] = 10 ** yh
    return out


def truth(sub, t_exec, metric):
    out = {}
    for el, s in sub.items():
        ia = np.searchsorted(s["t_all"], t_exec, side="right")
        if ia < len(s["t_all"]) and s["t_all"][ia] - t_exec <= TRUTH_DAYS:
            out[el] = BROKEN_ERR if s["broken"][ia] else s["v_all"][ia]
            continue
        i = np.searchsorted(s["t"], t_exec, side="right") - 1
        if i >= 0:
            out[el] = s["v"][i]
    return out


# ---------------------------------------------------------------- circuits
SIZES = {"small": dict(bv=(4, 8, 12), ghz=(4, 8, 12), qft=(4, 6, 8), adder=True),
         "large": dict(bv=(20, 30, 50), ghz=(20, 30, 50), qft=(12, 16), adder=False)}


def circuits(size="small"):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFTGate
    cfg = SIZES[size] if size != "all" else {k: tuple(SIZES["small"][k]) + tuple(SIZES["large"][k])
                                              if k != "adder" else True for k in SIZES["small"]}
    cs = {}
    for n in cfg["bv"]:
        qc = QuantumCircuit(n, n - 1, name=f"bv{n}")
        secret = [int(i % 3 != 1) for i in range(n - 1)]
        qc.x(n - 1)
        qc.h(range(n))
        for i, b in enumerate(secret):
            if b:
                qc.cx(i, n - 1)
        qc.h(range(n - 1))
        qc.measure(range(n - 1), range(n - 1))
        cs[qc.name] = qc
    for n in cfg["ghz"]:
        qc = QuantumCircuit(n, n, name=f"ghz{n}")
        qc.h(0)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        qc.measure(range(n), range(n))
        cs[qc.name] = qc
    for n in cfg["qft"]:
        qc = QuantumCircuit(n, n, name=f"qft{n}")
        qc.x([i for i in range(n) if i % 3 != 1])
        qc.append(QFTGate(n), range(n))
        qc.append(QFTGate(n).inverse(), range(n))
        qc.measure(range(n), range(n))
        cs[qc.name] = qc
    if cfg["adder"]:
        qc = QuantumCircuit(4, 2, name="adder4")      # 1-bit full adder: a, b, cin, cout
        qc.x([0, 1])
        qc.ccx(0, 1, 3)
        qc.cx(0, 1)
        qc.ccx(1, 2, 3)
        qc.cx(1, 2)
        qc.measure([2, 3], [0, 1])
        cs[qc.name] = qc
    return cs


def ideal_probs(qc):
    from qiskit.quantum_info import Statevector
    meas = sorted(((qc.find_bit(ins.clbits[0]).index, qc.find_bit(ins.qubits[0]).index)
                   for ins in qc.data if ins.operation.name == "measure"))
    sv = Statevector(qc.remove_final_measurements(inplace=False))
    return sv.probabilities_dict(qargs=[q for _, q in meas])


# ---------------------------------------------------------------- targets / scoring
def gate2q_class(name):
    from qiskit.circuit.library import CXGate, CZGate, ECRGate
    return {"cz": CZGate, "ecr": ECRGate, "cx": CXGate}[name]


def make_target(n, e2, e1, ro, meta, qprops=None):
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import Measure, RZGate, SXGate, XGate
    from qiskit.transpiler import InstructionProperties, Target
    t = Target(num_qubits=n, qubit_properties=qprops)
    m1 = float(np.median(list(e1.values()))) if e1 else 3e-4
    mr = float(np.median(list(ro.values()))) if ro else 2e-2
    edges = {}
    for el, err in e2.items():
        a, b = map(int, el.split("-"))
        if a < n and b < n:
            p = InstructionProperties(error=float(min(err, BROKEN_ERR)), duration=meta["dur_2q"])
            edges[(a, b)] = p
            edges[(b, a)] = InstructionProperties(error=p.error, duration=p.duration)
    t.add_instruction(gate2q_class(meta["gate2q"])(), edges)
    for g in (SXGate(), XGate()):
        t.add_instruction(g, {(q,): InstructionProperties(error=float(e1.get(str(q), m1)),
                                                          duration=meta["dur_1q"]) for q in range(n)})
    t.add_instruction(RZGate(Parameter("a")), {(q,): InstructionProperties(error=0.0, duration=0.0)
                                               for q in range(n)})
    t.add_instruction(Measure(), {(q,): InstructionProperties(error=float(ro.get(str(q), mr)),
                                                              duration=meta["dur_meas"]) for q in range(n)})
    return t


def esp(tq, tr, gate2q):
    m1 = np.median(list(tr["1q"].values()))
    mr = np.median(list(tr["ro"].values()))
    m2 = np.median(list(tr["2q"].values()))
    logp, n2 = 0.0, 0
    for ins in tq.data:
        name = ins.operation.name
        qs = [tq.find_bit(q).index for q in ins.qubits]
        if name == gate2q:
            e = tr["2q"].get("-".join(map(str, sorted(qs))), m2)
            n2 += 1
        elif name in ("sx", "x"):
            e = tr["1q"].get(str(qs[0]), m1)
        elif name == "measure":
            e = tr["ro"].get(str(qs[0]), mr)
        else:
            continue
        logp += np.log1p(-min(e, 0.999))
    return float(np.exp(logp)), n2


def simulate(tq, tr, meta, ideal, shots, seed):
    """Noisy simulation on only the qubits the compiled circuit uses."""
    from qiskit import QuantumCircuit
    from qiskit.providers import BackendV2, Options, QubitProperties
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel

    class _TB(BackendV2):
        def __init__(self, target):
            super().__init__(name="truth")
            self._t = target

        @property
        def target(self):
            return self._t

        @property
        def max_circuits(self):
            return None

        @classmethod
        def _default_options(cls):
            return Options()

        def run(self, *a, **k):
            raise NotImplementedError

    active = sorted({tq.find_bit(q).index for ins in tq.data for q in ins.qubits
                     if ins.operation.name not in ("barrier", "delay")})
    mp = {p: i for i, p in enumerate(active)}
    small = QuantumCircuit(len(active), tq.num_clbits)
    pairs = set()
    for ins in tq.data:
        if ins.operation.name in ("barrier", "delay"):
            continue
        qs = [mp[tq.find_bit(q).index] for q in ins.qubits]
        if len(qs) == 2:
            pairs.add(tuple(qs))
        small.append(ins.operation, [small.qubits[i] for i in qs],
                     [small.clbits[tq.find_bit(c).index] for c in ins.clbits])
    mt1 = np.median(list(tr["t1"].values()))
    mt2 = np.median(list(tr["t2"].values()))
    qprops = []
    for p in active:
        t1 = tr["t1"].get(str(p), mt1) * 1e-6
        t2 = min(tr["t2"].get(str(p), mt2) * 1e-6, 1.98 * t1)
        qprops.append(QubitProperties(t1=t1, t2=t2, frequency=5e9))
    e2 = {}
    for a, b in pairs:
        key = "-".join(map(str, sorted((active[a], active[b]))))
        e2[f"{min(a, b)}-{max(a, b)}"] = tr["2q"].get(key, np.median(list(tr["2q"].values())))
    e1 = {str(i): tr["1q"].get(str(p), np.median(list(tr["1q"].values()))) for i, p in enumerate(active)}
    ro = {str(i): tr["ro"].get(str(p), np.median(list(tr["ro"].values()))) for i, p in enumerate(active)}
    tgt = make_target(len(active), e2, e1, ro, meta, qprops)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nm = NoiseModel.from_backend(_TB(tgt))
        counts = AerSimulator(noise_model=nm).run(small, shots=shots, seed_simulator=seed).result().get_counts()
    keys = set(counts) | set(ideal)
    tvd = 0.5 * sum(abs(counts.get(k, 0) / shots - ideal.get(k, 0.0)) for k in keys)
    return 1 - tvd


# ---------------------------------------------------------------- worker
G = {}


def _init(S, meta, args):
    G.update(S=S, meta=meta, args=args, circs=circuits(args.circuits))
    small = set(circuits("small"))                  # full simulation only for the small set
    G["ideal"] = {k: ideal_probs(c) for k, c in G["circs"].items() if k in small}


def run_day(job):
    from qiskit import transpile
    day, do_sim = job
    S, meta, a = G["S"], G["meta"], G["args"]
    t_exec = day + 0.5
    tr = {m: truth(S[m], t_exec, m) for m in ("2q", "1q", "ro", "t1", "t2") if m in S}
    esp_rows, sim_rows = [], []
    for D in a.delays:
        t_cut = t_exec - D / 24
        tgts = {}
        for meth in METHODS:
            maps = {m: predict(S[m], t_cut, meth, a.window) for m in ("2q", "1q", "ro")}
            tgts[meth] = make_target(meta["n_qubits"], maps["2q"], maps["1q"], maps["ro"], meta)
        for ci, (cname, qc) in enumerate(G["circs"].items()):
            seed = int(int(day) * 97 + ci * 13 + round(D * 60)) % (2 ** 31)
            layouts = {}
            for meth in METHODS:
                tq = transpile(qc, target=tgts[meth], optimization_level=3, seed_transpiler=seed)
                p, n2 = esp(tq, tr, meta["gate2q"])
                used = tuple(sorted({tq.find_bit(q).index for ins in tq.data for q in ins.qubits}))
                layouts[meth] = used
                esp_rows.append(dict(day=day, delay=D, circuit=cname, method=meth, esp=p, n2q=n2,
                                     depth=tq.depth(), qubits=" ".join(map(str, used))))
                if do_sim and cname in G["ideal"]:
                    f = simulate(tq, tr, meta, G["ideal"][cname], a.shots, seed)
                    sim_rows.append(dict(day=day, delay=D, circuit=cname, method=meth, fidelity=f, esp=p))
    return esp_rows, sim_rows


# ---------------------------------------------------------------- summary
def boot_ci(per_day, n=2000, seed=0):
    x = np.asarray(per_day)
    bs = np.random.default_rng(seed).choice(x, size=(n, len(x))).mean(axis=1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def wtl(a, b, tol=1e-9):
    """win / tie / loss of a vs b in % of runs (tie = same score, e.g. identical layout)."""
    d = (a - b).to_numpy()
    return f"{(d > tol).mean():.0%}/{(abs(d) <= tol).mean():.0%}/{(d < -tol).mean():.0%}"


def summarize(df, score, label):
    """Paired gain of asym over lcd and over w10, in % of score; CI over days."""
    pv = df.pivot_table(index=["day", "delay", "circuit"], columns="method", values=score).dropna()
    rows = []
    for (D, c), g in pv.groupby(level=["delay", "circuit"]):
        r = {"delay_h": D, "circuit": c, **{f"mean_{m}": g[m].mean() for m in METHODS}}
        for ref in ("lcd", "w10"):
            lr = np.log(g["asym"].clip(lower=1e-12) / g[ref].clip(lower=1e-12))
            m_, lo, hi = boot_ci(lr.to_numpy())
            r[f"gain_vs_{ref}_%"] = 100 * (np.exp(m_) - 1)
            r[f"ci_vs_{ref}"] = f"[{100 * (np.exp(lo) - 1):+.1f}, {100 * (np.exp(hi) - 1):+.1f}]"
            r[f"win_vs_{ref}"] = wtl(g["asym"], g[ref])
        rows.append(r)
    for D, g in pv.groupby(level="delay"):
        r = {"delay_h": D, "circuit": "ALL", **{f"mean_{m}": g[m].mean() for m in METHODS}}
        for ref in ("lcd", "w10"):
            lr = np.log(g["asym"].clip(lower=1e-12) / g[ref].clip(lower=1e-12)).groupby(level="day").mean()
            m_, lo, hi = boot_ci(lr.to_numpy())
            r[f"gain_vs_{ref}_%"] = 100 * (np.exp(m_) - 1)
            r[f"ci_vs_{ref}"] = f"[{100 * (np.exp(lo) - 1):+.1f}, {100 * (np.exp(hi) - 1):+.1f}]"
            r[f"win_vs_{ref}"] = wtl(g["asym"], g[ref])
        rows.append(r)
    t = pd.DataFrame(rows)
    print(f"\n--- {label}: gain of asym (geometric mean ratio, 95% CI over days; win/tie/loss = % of runs asym is higher / equal / lower)")
    cols = ["delay_h", "circuit"] + [f"mean_{m}" for m in METHODS] + \
           ["gain_vs_lcd_%", "ci_vs_lcd", "win_vs_lcd", "gain_vs_w10_%", "ci_vs_w10", "win_vs_w10"]
    print(t[cols].round(3).to_string(index=False))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machines", nargs="*")
    ap.add_argument("--delays", nargs="*", type=float, default=[0, 6, 12])
    ap.add_argument("--day-step", type=int, default=2, help="use every N-th test day")
    ap.add_argument("--full-sim-every", type=int, default=5, help="full noisy simulation on every N-th used day (0 = never)")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--h-down", type=float, default=5)
    ap.add_argument("--h-up", type=float, default=10)
    ap.add_argument("--window", type=float, default=10)
    ap.add_argument("--circuits", choices=["small", "large", "all"], default="small",
                    help="small: 4-12 qubits (default); large: 12-50 qubits, ESP only (no full simulation)")
    ap.add_argument("--tag", default="", help="suffix for output files, e.g. _large")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    if a.circuits != "small" and not a.tag:
        a.tag = f"_{a.circuits}"
    OUT.mkdir(exist_ok=True)
    files = sorted(Path("data/standard").glob("*_cal.csv.gz"))
    if a.machines:
        files = [f for f in files if f.name.replace("_cal.csv.gz", "") in a.machines]
    if not files:
        raise SystemExit("No data/standard/*_cal.csv.gz found. Run adapter_calibration.py first.")

    all_esp, all_sim = {}, {}
    for f in files:
        machine = f.name.replace("_cal.csv.gz", "")
        meta = json.loads((f.parent / f"{machine}_meta.json").read_text())
        cal = pd.read_csv(f, dtype={"element": str})
        cal["t"] = pd.to_datetime(cal.t, utc=True, format="mixed")
        S = build_series(cal, a.h_down, a.h_up)
        t2 = cal[cal.metric == "2q"].t
        split = t2.min() + (t2.max() - t2.min()) / 2
        d0 = int(np.ceil((split - EPOCH).total_seconds() / 86400))
        d1 = int(np.floor((t2.max() - EPOCH).total_seconds() / 86400)) - TRUTH_DAYS
        days = list(range(d0, d1, a.day_step))
        jobs = [(d, a.circuits != "large" and a.full_sim_every > 0 and k % a.full_sim_every == 0)
                for k, d in enumerate(days)]
        print(f"\n===== {machine} =====  {len(days)} test days from {split:%Y-%m-%d}, "
              f"{sum(s for _, s in jobs)} with full simulation, delays {a.delays} h, {a.workers} workers")
        esp_rows, sim_rows = [], []
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(S, meta, a)) as ex:
            for k, (er, sr) in enumerate(ex.map(run_day, jobs)):
                esp_rows += er
                sim_rows += sr
                if (k + 1) % 10 == 0 or k + 1 == len(jobs):
                    print(f"  {k + 1}/{len(jobs)} days done")
        e = pd.DataFrame(esp_rows)
        e["date"] = (EPOCH + pd.to_timedelta(e.day, unit="D")).dt.strftime("%Y-%m-%d")
        e.to_csv(OUT / f"esp_{machine}{a.tag}.csv", index=False)
        lay = e.pivot_table(index=["day", "delay", "circuit"], columns="method", values="qubits", aggfunc="first")
        print(f"  compiler sanity: physical qubits differ between lcd and asym in "
              f"{(lay['lcd'] != lay['asym']).mean():.0%} of compilations, lcd vs w10 in "
              f"{(lay['lcd'] != lay['w10']).mean():.0%}")
        all_esp[machine] = summarize(e, "esp", f"{machine} ESP")
        if sim_rows:
            s = pd.DataFrame(sim_rows)
            s.to_csv(OUT / f"sim_{machine}{a.tag}.csv", index=False)
            all_sim[machine] = summarize(s, "fidelity", f"{machine} simulated fidelity")
            pv_e = s.pivot_table(index=["day", "delay", "circuit"], columns="method", values="esp")
            pv_f = s.pivot_table(index=["day", "delay", "circuit"], columns="method", values="fidelity")
            same = (np.sign(pv_e["asym"] - pv_e["lcd"]) == np.sign(pv_f["asym"] - pv_f["lcd"])).mean()
            rho = s[["esp", "fidelity"]].corr(method="spearman").iloc[0, 1]
            print(f"  ESP vs simulated fidelity: rank correlation {rho:.2f}; "
                  f"same winner (asym vs lcd) in {same:.0%} of runs")

    # --- plot: ESP gain of asym vs lcd and vs w10, per circuit and delay
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    machines = list(all_esp)
    fig, axes = plt.subplots(2, len(machines), figsize=(5.2 * len(machines), 7), squeeze=False)
    for j, m in enumerate(machines):
        t = all_esp[m]
        circs = [c for c in t.circuit.unique() if c != "ALL"] + ["ALL"]
        x = np.arange(len(circs))
        w = 0.8 / len(a.delays)
        for i, ref in enumerate(("lcd", "w10")):
            ax = axes[i, j]
            for k, D in enumerate(a.delays):
                v = [t[(t.delay_h == D) & (t.circuit == c)][f"gain_vs_{ref}_%"].iloc[0] for c in circs]
                ax.bar(x + (k - (len(a.delays) - 1) / 2) * w, v, w, label=f"delay {D:g} h")
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(x, circs, rotation=45, fontsize=8)
            ax.set_ylabel(f"ESP gain of asym vs {ref} (%)")
            if i == 0:
                ax.set_title(m)
            ax.legend(fontsize=7)
    fig.suptitle("Closed loop, estimated success probability\nasym vs lcd (top), asym vs w10 (bottom); above 0 = asym better", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / f"gain{a.tag}.png", dpi=130)
    print(f"\nSaved {OUT}/esp_<machine>{a.tag}.csv, {OUT}/sim_<machine>{a.tag}.csv, {OUT}/gain{a.tag}.png")


if __name__ == "__main__":
    main()
