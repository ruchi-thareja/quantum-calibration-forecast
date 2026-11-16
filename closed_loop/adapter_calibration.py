"""
Adapter for the closed-loop simulation: everything the compiler and the noise model need.

Run:  python3 adapter_calibration.py
Reads data/tidy/gates.csv and data/tidy/qubits.csv. Writes, per backend:
  data/standard/<backend>_cal.csv.gz   columns metric, element, t, value
      metric: 2q (2Q gate error, link "a-b"; value >= 1 kept = IBM's "broken link" flag)
              1q (sx gate error, qubit), ro (readout error, qubit), t1, t2 (microseconds, qubit)
      t = when IBM measured it, rounded to the hour; stale repeats removed
  data/standard/<backend>_meta.json    2Q gate name, number of qubits, typical gate durations (s)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

TIDY = Path("data/tidy")
OUT = Path("data/standard")
UNIT = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0}


def load(path, cols, keep):
    parts = []
    for ch in pd.read_csv(path, chunksize=500_000, usecols=cols, dtype={"qubits": str}):
        parts.append(ch[keep(ch)])
    return pd.concat(parts, ignore_index=True)


def tidy_time(df):
    df["t"] = pd.to_datetime(df.measured_at.fillna(df.snapshot), utc=True, format="mixed",
                             errors="coerce").dt.floor("h")
    df["value"] = pd.to_numeric(df.value, errors="coerce")
    return df.dropna(subset=["t", "value"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = load(TIDY / "gates.csv", ["backend", "snapshot", "gate", "qubits", "param", "value", "unit", "measured_at"],
             lambda c: c.param.isin(["gate_error", "gate_length"]))
    q = load(TIDY / "qubits.csv", ["backend", "snapshot", "qubit", "param", "value", "unit", "measured_at"],
             lambda c: c.param.isin(["T1", "T2", "readout_error", "readout_length"]))
    g, q = tidy_time(g), tidy_time(q)
    g["two"] = g.qubits.astype(str).str.contains("-")

    for backend in sorted(set(g.backend) | set(q.backend)):
        gb, qb = g[g.backend == backend], q[q.backend == backend]
        g2 = gb[gb.two]
        gate2q = g2[g2.param == "gate_error"].gate.mode().iloc[0]
        rows = []

        e2 = g2[(g2.gate == gate2q) & (g2.param == "gate_error")].copy()
        e2["element"] = e2.qubits.apply(lambda s: "-".join(sorted(s.split("-"), key=int)))
        e2 = e2[e2.value > 0]
        rows.append(e2.groupby(["element", "t"], as_index=False).value.max().assign(metric="2q"))

        e1 = gb[(~gb.two) & (gb.gate == "sx") & (gb.param == "gate_error")].copy()
        e1["element"] = e1.qubits.astype(str)
        e1 = e1[(e1.value > 0) & (e1.value < 1)]
        rows.append(e1.groupby(["element", "t"], as_index=False).value.mean().assign(metric="1q"))

        for param, metric in [("readout_error", "ro"), ("T1", "t1"), ("T2", "t2")]:
            x = qb[qb.param == param].copy()
            x["element"] = x.qubit.astype(int).astype(str)
            x = x[(x.value > 0) & ((x.value < 1) if metric == "ro" else (x.value > 0))]
            if metric in ("t1", "t2"):   # store in microseconds
                x["value"] = x.value * x.unit.map(UNIT).fillna(1e-6) / 1e-6
            rows.append(x.groupby(["element", "t"], as_index=False).value.mean().assign(metric=metric))

        cal = pd.concat(rows, ignore_index=True)[["metric", "element", "t", "value"]]
        cal.to_csv(OUT / f"{backend}_cal.csv.gz", index=False)

        def dur(df):
            d = df[df.param.isin(["gate_length", "readout_length"])]
            return float(np.nanmedian(d.value * d.unit.map(UNIT).fillna(1e-9))) if len(d) else None

        meta = {"gate2q": gate2q,
                "n_qubits": int(max(qb.qubit.max() if len(qb) else 0,
                                    e2.element.str.split("-").explode().astype(int).max())) + 1,
                "dur_2q": dur(g2[g2.gate == gate2q]) or 68e-9,
                "dur_1q": dur(gb[(~gb.two) & (gb.gate == "sx")]) or 32e-9,
                "dur_meas": dur(qb[qb.param == "readout_length"]) or 1.5e-6}
        (OUT / f"{backend}_meta.json").write_text(json.dumps(meta, indent=1))
        counts = cal.groupby("metric").size().to_dict()
        print(f"{backend}: 2Q gate '{gate2q}', {meta['n_qubits']} qubits, rows {counts}, "
              f"broken-link flags {(e2.value >= 1).sum()}, durations 2Q {meta['dur_2q']*1e9:.0f} ns, "
              f"1Q {meta['dur_1q']*1e9:.0f} ns, readout {meta['dur_meas']*1e6:.2f} us")


if __name__ == "__main__":
    main()
