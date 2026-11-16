"""
Adapter: IBM calibration data -> standard table (the only IBM-specific step).

Run:  python3 adapter_ibm.py
Reads data/tidy/gates.csv, writes data/standard/<backend>_2q.csv.gz with columns:
    element, t, value
  element = a link like "12-13" (direction merged)
  t       = when IBM actually MEASURED it (not when we downloaded it)
  value   = 2Q gate error

What it cleans:
  - stale values: if a link wasn't remeasured, IBM repeats the old value with the old
    measurement time -> we keep it once, so "no change" isn't counted as a prediction
  - both directions of a link (12-13 and 13-12) -> merged into one measurement
  - error = 1 or <= 0 (IBM's way of marking a broken / disabled link) -> dropped, counted
"""
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path("data/tidy/gates.csv")
OUT = Path("data/standard")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parts = []
    for chunk in pd.read_csv(SRC, chunksize=500_000,
                             usecols=["backend", "snapshot", "qubits", "param", "value", "measured_at"]):
        keep = (chunk.param == "gate_error") & chunk.qubits.astype(str).str.contains("-")
        parts.append(chunk[keep])
    df = pd.concat(parts, ignore_index=True)

    df["t"] = pd.to_datetime(df.measured_at.fillna(df.snapshot), utc=True, format="mixed", errors="coerce")
    df["value"] = pd.to_numeric(df.value, errors="coerce")
    df = df.dropna(subset=["t", "value"])
    df["element"] = df.qubits.astype(str).apply(lambda s: "-".join(sorted(s.split("-"), key=int)))
    # the two directions of a link can differ by a few seconds -> round to the hour
    df["t"] = df.t.dt.floor("h")

    for backend, g in df.groupby("backend"):
        n_rows = len(g)
        broken = ((g.value <= 0) | (g.value >= 1)).sum()
        g = g[(g.value > 0) & (g.value < 1)]
        m = g.groupby(["element", "t"], as_index=False).value.mean()  # one row per real measurement
        m = m.sort_values(["element", "t"])
        gaps = m.groupby("element").t.diff().dt.total_seconds() / 86400
        m.to_csv(OUT / f"{backend}_2q.csv.gz", index=False)
        print(f"{backend}: {n_rows} rows -> {len(m)} real measurements "
              f"({1 - len(m) / max(n_rows, 1):.0%} were stale repeats or duplicates), "
              f"{broken} broken-link values dropped, {m.element.nunique()} links, "
              f"median time between remeasurements = {np.nanmedian(gaps):.2f} days")


if __name__ == "__main__":
    main()
