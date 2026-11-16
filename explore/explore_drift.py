"""
First look at how 2Q gate errors drift over time.

Run:  python3 explore_drift.py ibm_fez
(needs: pip3 install pandas matplotlib)

Makes, in the plots/ folder:
  <backend>_pair_summary.csv  - one row per qubit pair: median error, spread, how "jumpy" it is
  <backend>_drift.png         - error over the year for 6 representative pairs
  <backend>_daily_change.png  - how big day-to-day changes are (shape of the drift)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

backend = sys.argv[1] if len(sys.argv) > 1 else "ibm_fez"
out = Path("plots")
out.mkdir(exist_ok=True)

# --- load only what we need (the CSV is huge, so read in chunks and filter) ---
parts = []
for chunk in pd.read_csv("data/tidy/gates.csv", chunksize=500_000,
                         usecols=["backend", "snapshot", "gate", "qubits", "param", "value"]):
    keep = (chunk.backend == backend) & (chunk.param == "gate_error") & chunk.qubits.str.contains("-")
    parts.append(chunk[keep])
df = pd.concat(parts)
if df.empty:
    sys.exit(f"No 2Q gate_error rows for {backend}. Check the backend name.")

print("2Q gate types found:", df.gate.unique().tolist())
df["t"] = pd.to_datetime(df.snapshot, utc=True, format="mixed")
df["value"] = pd.to_numeric(df.value, errors="coerce")

# IBM sometimes lists a pair in both directions (0-1 and 1-0): treat as one link
df["pair"] = df.qubits.apply(lambda s: "-".join(sorted(s.split("-"), key=int)))
# IBM reports error = 1 for broken/disabled links; keep them out of the stats
df = df[(df.value > 0) & (df.value < 1)]
ts = df.groupby(["pair", "t"]).value.mean().unstack("pair").sort_index()

print(f"{ts.shape[1]} qubit pairs, {ts.shape[0]} snapshots, "
      f"{ts.index.min():%Y-%m-%d} -> {ts.index.max():%Y-%m-%d}")

# --- per-pair summary ---
rel_change = (ts.diff().abs() / ts.shift()).median()   # typical day-to-day relative change
summary = pd.DataFrame({
    "median_error": ts.median(),
    "std": ts.std(),
    "cv": ts.std() / ts.mean(),                           # spread relative to the mean
    "typical_daily_rel_change": rel_change,
    "max_over_median": ts.max() / ts.median(),            # size of the worst spike
    "n_snapshots": ts.count(),
}).sort_values("cv")
summary.to_csv(out / f"{backend}_pair_summary.csv")

# --- pick 6 representative pairs: 2 stable, 2 typical, 2 volatile ---
good = summary[summary.n_snapshots > 0.8 * len(ts)]
idx = [0, 1, len(good) // 2, len(good) // 2 + 1, len(good) - 2, len(good) - 1]
picks = good.index[idx]

fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
for ax, p, label in zip(axes.flat, picks, ["stable"] * 2 + ["typical"] * 2 + ["volatile"] * 2):
    s = ts[p].dropna()
    ax.plot(s.index, s.values, lw=0.8)
    ax.set_yscale("log")
    ax.set_title(f"pair {p} ({label}, CV={summary.loc[p, 'cv']:.2f})", fontsize=10)
    ax.set_ylabel("2Q gate error")
fig.suptitle(f"{backend}: 2Q gate error over time")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(out / f"{backend}_drift.png", dpi=130)

# --- shape of the day-to-day changes: smooth drift or sudden jumps? ---
ratios = []
for p in ts.columns:
    s = ts[p].dropna()            # compare each snapshot with that pair's previous one
    ratios.append((s / s.shift()).dropna())
logv = np.log10(pd.concat(ratios).to_numpy())
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(logv, bins=200, range=(-1, 1))
ax.set_yscale("log")
ax.set_xlabel("log10(error today / error previous snapshot)")
ax.set_ylabel("count (log scale)")
ax.set_title(f"{backend}: day-to-day changes, all pairs")
fig.tight_layout()
fig.savefig(out / f"{backend}_daily_change.png", dpi=130)

big = (np.abs(logv) > np.log10(2)).mean()
print(f"Share of day-to-day changes bigger than 2x: {big:.1%}")
print(f"Saved plots and summary in {out}/")
