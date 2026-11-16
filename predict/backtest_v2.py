"""
Backtest v2: lcd, avg, w-n sweep, EWMA (finer h), asymmetric EWMA (h_down, h_up).

Run (after adapter_ibm.py):  python3 backtest_v2.py
Same rules as v1: log10 errors, walk-forward (only past data), first half of each
machine's history = tuning, second half = test.

New in v2
  - w-n swept over several windows (days), so the paper's w-15 is no longer a fixed guess.
  - EWMA half-life grid restricted to the useful range found in v1 (1-10 cycles).
  - Asymmetric EWMA: when the new value is LOWER than the estimate it updates with
    half-life h_down, when HIGHER with h_up. h_down < h_up = "follow fixes fast, ignore
    spikes". h_down == h_up is plain EWMA, so the grid contains EWMA as a special case.
  - Significance tests against the best w-n (the real competitor), not only lcd,
    on error AND on top-20% overlap.
  - Tuning criterion selectable (TUNE_METRIC).

Outputs (results_v2/)
  curves.png        tuning period: error / ranking / top-20% vs window and half-life
  asym_heatmap.png  tuning period: asymmetric EWMA over (h_down, h_up)
  daily_<m>.csv     per-day scores of every method (for later analysis)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

W_GRID = [3, 5, 7, 10, 15, 20, 30]                 # window lengths (days)
H_GRID = [1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10]         # EWMA half-lives (calibration cycles)
H_DOWN = [0.5, 1, 2, 3, 5]                         # asymmetric: half-life when value drops
H_UP = [3, 5, 7, 10, 15, 30]                       # asymmetric: half-life when value rises
TUNE_METRIC = "mae"        # "mae" (lower better), "spearman" or "top20" (higher better)
MIN_POINTS = 30
MIN_LINKS_PER_DAY = 20
TOP_FRAC = 0.2
OUT = Path("results_v2")

W_COLS = [f"w{w}" for w in W_GRID]
E_COLS = [f"ewma_h{h:g}" for h in H_GRID]
PAIRS = [(d, u) for d in H_DOWN for u in H_UP]
A_COLS = [f"asym_d{d:g}_u{u:g}" for d, u in PAIRS]
ALL_COLS = ["lcd", "avg"] + W_COLS + E_COLS + A_COLS


def predict_element(y, td, cycle):
    n = len(y)
    i = np.arange(1, n)
    cs = np.concatenate([[0.0], np.cumsum(y)])
    cols = {"lcd": y[:-1], "avg": cs[i] / i}
    for w in W_GRID:
        j0 = np.searchsorted(td, td[i] - w, side="left")
        cnt = i - j0
        cols[f"w{w}"] = np.where(cnt > 0, (cs[i] - cs[j0]) / np.maximum(cnt, 1), y[:-1])

    H = np.array(H_GRID, dtype=float)
    HD = np.array([d for d, _ in PAIRS], dtype=float)
    HU = np.array([u for _, u in PAIRS], dtype=float)
    m = np.full(len(H), y[0])
    ma = np.full(len(PAIRS), y[0])
    E = np.empty((n - 1, len(H)))
    A = np.empty((n - 1, len(PAIRS)))
    for k in range(1, n):
        E[k - 1], A[k - 1] = m, ma                    # prediction made BEFORE seeing y[k]
        dc = (td[k] - td[k - 1]) / cycle              # elapsed calibration cycles
        lam = 0.5 ** (dc / H)
        m = lam * m + (1 - lam) * y[k]
        lam_a = 0.5 ** (dc / np.where(y[k] < ma, HD, HU))
        ma = lam_a * ma + (1 - lam_a) * y[k]
    for c, col in zip(E_COLS, E.T):
        cols[c] = col
    for c, col in zip(A_COLS, A.T):
        cols[c] = col
    return cols


def predict_machine(df):
    df = df.sort_values(["element", "t"])
    cycle = float(np.nanmedian(df.groupby("element").t.diff().dt.total_seconds() / 86400))
    blocks = []
    for el, g in df.groupby("element"):
        if len(g) < MIN_POINTS:
            continue
        y = np.log10(g.value.to_numpy())
        td = (g.t - g.t.iloc[0]).dt.total_seconds().to_numpy() / 86400
        b = pd.DataFrame(predict_element(y, td, cycle))
        b.insert(0, "y", y[1:])
        b.insert(0, "t", g.t.iloc[1:].to_numpy())
        b.insert(0, "element", el)
        blocks.append(b)
    return pd.concat(blocks, ignore_index=True), cycle


def daily_metrics(p):
    """Per-day MAE, Spearman and top-20% overlap for every method."""
    day = p.t.dt.floor("D")
    mae = p[ALL_COLS].sub(p.y, axis=0).abs().groupby(day).mean()
    sp, top, days = [], [], []
    for d, g in p.groupby(day):
        if len(g) < MIN_LINKS_PER_DAY:
            continue
        sp.append(g[ALL_COLS].rank().corrwith(g.y.rank()))
        k = max(1, int(TOP_FRAC * len(g)))
        true_best = set(g.element.values[np.argsort(g.y.values, kind="stable")[:k]])
        top.append(pd.Series({c: len(set(g.element.values[np.argsort(g[c].values, kind="stable")[:k]])
                                      & true_best) / k for c in ALL_COLS}))
        days.append(d)
    sp = pd.DataFrame(sp, index=days)
    top = pd.DataFrame(top, index=days)
    return {"mae": mae.loc[days], "spearman": sp, "top20": top}


def pick_best(scores, cols):
    s = scores.loc[cols, TUNE_METRIC]
    return s.idxmin() if TUNE_METRIC == "mae" else s.idxmax()


def boot(a, b, n_boot=2000, seed=0):
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def verdict(lo, hi, lower_is_better):
    if hi < 0:
        return "A better" if lower_is_better else "B better"
    if lo > 0:
        return "B better" if lower_is_better else "A better"
    return "no significant difference"


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(Path("data/standard").glob("*_2q.csv.gz"))
    if not files:
        raise SystemExit("No data/standard/*_2q.csv.gz found. Run adapter_ibm.py first.")

    tune_scores, daily, split_day = {}, {}, {}
    for f in files:
        machine = f.name.replace("_2q.csv.gz", "")
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df.t, utc=True, format="mixed")
        p, cycle = predict_machine(df)
        dm = daily_metrics(p)
        days = dm["mae"].index
        split = days.min() + (days.max() - days.min()) / 2
        split_day[machine], daily[machine] = split, dm
        tune_scores[machine] = pd.DataFrame({k: v[v.index < split].mean() for k, v in dm.items()})
        long = pd.concat({k: v for k, v in dm.items()}, axis=1)
        long.to_csv(OUT / f"daily_{machine}.csv")
        print(f"{machine}: {p.element.nunique()} links, {len(p)} predictions, cycle = {cycle:.2f} days, "
              f"split at {split:%Y-%m-%d}")

    machines = list(daily)
    ref = "ibm_fez" if "ibm_fez" in machines else machines[0]
    best = {m: {"w": pick_best(tune_scores[m], W_COLS), "ewma": pick_best(tune_scores[m], E_COLS),
                "asym": pick_best(tune_scores[m], A_COLS)} for m in machines}
    print(f"\nTuned on first half by '{TUNE_METRIC}':")
    for m in machines:
        print(f"  {m}: best window {best[m]['w']}, best EWMA {best[m]['ewma']}, best asymmetric {best[m]['asym']}")

    print("\n=== TEST period (second half, never used for tuning) ===")
    print("mae: lower is better | spearman, top20: higher is better")
    for m in machines:
        dm = daily[m]
        test = {k: v[v.index >= split_day[m]] for k, v in dm.items()}
        cols = ["lcd", "avg", best[m]["w"], best[m]["ewma"], best[m]["asym"]]
        labels = ["lcd", "avg", f"{best[m]['w']} (best window)", f"{best[m]['ewma']} (best EWMA)",
                  f"{best[m]['asym']} (best asym)"]
        if m != ref:
            for fam in ("w", "ewma", "asym"):
                if best[ref][fam] not in cols:
                    cols.append(best[ref][fam])
                    labels.append(f"{best[ref][fam]} (tuned on {ref})")
        tab = pd.DataFrame({k: test[k][cols].mean().to_numpy() for k in ("mae", "spearman", "top20")},
                           index=labels)
        print(f"\n{m}")
        print(tab.round(4).to_string())

        W, E, A = best[m]["w"], best[m]["ewma"], best[m]["asym"]
        checks = [
            ("asym vs best window", A, W, "mae", True), ("asym vs best window", A, W, "top20", False),
            ("asym vs lcd", A, "lcd", "top20", False),
            ("EWMA vs best window", E, W, "mae", True), ("best window vs lcd", W, "lcd", "mae", True),
        ]
        print("  significance (bootstrap over days, 95% CI of A - B):")
        for name, a, b, metric, lower in checks:
            d, lo, hi = boot(test[metric][a], test[metric][b])
            print(f"    {name:22s} {metric:6s}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {verdict(lo, hi, lower)}")

    # --- curves (tuning period) ---
    fig, axes = plt.subplots(3, len(machines), figsize=(4.6 * len(machines), 10), squeeze=False)
    names = {"mae": "mean |log10 error| (lower better)", "spearman": "rank correlation (higher better)",
             "top20": "top-20% overlap (higher better)"}
    for j, m in enumerate(machines):
        s = tune_scores[m]
        for i, metric in enumerate(["mae", "spearman", "top20"]):
            ax = axes[i, j]
            ax.plot(W_GRID, s.loc[W_COLS, metric], "s-", color="tab:green", label="window w-n (n in days)")
            ax.plot(H_GRID, s.loc[E_COLS, metric], "o-", color="tab:blue", label="EWMA (h in cycles)")
            ax.axhline(s.loc["lcd", metric], color="tab:red", ls="--", label="lcd")
            ax.axhline(s.loc[best[m]["asym"], metric], color="tab:purple", ls=":", label=f"best asym")
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 3, 5, 10, 20, 30], ["1", "2", "3", "5", "10", "20", "30"])
            ax.set_ylabel(names[metric], fontsize=8)
            if i == 0:
                ax.set_title(m)
            if i == 2:
                ax.set_xlabel("window (days) / half-life (cycles)")
            ax.legend(fontsize=7)
    fig.suptitle("Tuning period")
    fig.tight_layout()
    fig.savefig(OUT / "curves.png", dpi=130)

    # --- asymmetric heatmaps (tuning period) ---
    fig, axes = plt.subplots(2, len(machines), figsize=(4.6 * len(machines), 7.5), squeeze=False)
    for j, m in enumerate(machines):
        s = tune_scores[m]
        for i, metric in enumerate(["mae", "top20"]):
            grid = np.array([[s.loc[f"asym_d{d:g}_u{u:g}", metric] for u in H_UP] for d in H_DOWN])
            ax = axes[i, j]
            im = ax.imshow(grid, cmap="viridis_r" if metric == "mae" else "viridis", aspect="auto")
            ax.set_xticks(range(len(H_UP)), [f"{u:g}" for u in H_UP])
            ax.set_yticks(range(len(H_DOWN)), [f"{d:g}" for d in H_DOWN])
            ax.set_xlabel("h_up (when error rises)")
            ax.set_ylabel("h_down (when error drops)")
            lo, hi = np.nanmin(grid), np.nanmax(grid)
            for a in range(len(H_DOWN)):
                for b in range(len(H_UP)):
                    frac = (grid[a, b] - lo) / (hi - lo + 1e-12)
                    bright = frac < 0.5 if metric == "mae" else frac > 0.5   # yellow end of the colormap
                    ax.text(b, a, f"{grid[a, b]:.3f}", ha="center", va="center", fontsize=7,
                            color="black" if bright else "white")
            ax.set_title(f"{m}: {metric}" + (" (lower better)" if metric == "mae" else " (higher better)"),
                         fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Tuning period: asymmetric EWMA. Cells where h_down = h_up are plain EWMA.")
    fig.tight_layout()
    fig.savefig(OUT / "asym_heatmap.png", dpi=130)
    print(f"\nSaved {OUT}/curves.png, {OUT}/asym_heatmap.png, {OUT}/daily_<machine>.csv")


if __name__ == "__main__":
    main()
