"""
Diagnostic: do jumps in 2Q error persist or revert? How long do spikes last?

Run (after adapter_ibm.py):  python3 diagnostic.py
Uses every data/standard/*_2q.csv.gz (only real remeasurements, stale repeats removed).

Question 1 - persistence
  For every jump between two consecutive calibrations bigger than 1.5x, 2x or 3x:
  how much of the jump is still there 1 and 3 calibrations later?
    "persisted" = at least half of the jump is still there.
  If DROPS persist and RISES revert -> asymmetric EWMA (follow drops fast, resist rises).

Question 2 - duration
  Baseline = median of the previous 7 calibrations (only past data).
  Spike = value > 2x baseline; it ends when the value is back below 1.5x baseline.
  Dip   = value < baseline/2;  it ends when back above baseline/1.5.
  Duration = number of calibrations it lasts. Tells how much averaging is needed
  to wash spikes out, i.e. a physical reason for the memory length.

Outputs: plots/diagnostic_<machine>.png and a printed summary.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THRESHOLDS = [1.5, 2, 3]
BASE_WINDOW = 7
MAX_DUR = 10
OUT = Path("plots")


def persistence(series_list):
    """Fraction of rises / drops that still keep >= half the jump after 1 and 3 calibrations."""
    res = {}
    for thr in THRESHOLDS:
        L = np.log10(thr)
        acc = {("rise", 1): [], ("rise", 3): [], ("drop", 1): [], ("drop", 3): []}
        for y in series_list:
            if len(y) < 5:
                continue
            jump = y[1:-3] - y[:-4]               # jump at calibration i (i = 1 .. n-4)
            before = y[:-4]
            for lag in (1, 3):
                after = y[1 + lag: len(y) - 3 + lag]
                kept = (after - before) / np.where(jump == 0, np.nan, jump)
                acc[("rise", lag)].append(kept[jump > L] >= 0.5)
                acc[("drop", lag)].append(kept[jump < -L] >= 0.5)
        for key, parts in acc.items():
            a = np.concatenate(parts) if parts else np.array([])
            res[(thr,) + key] = (a.mean() if len(a) else np.nan, len(a))
    return res


def durations(series_list):
    spikes, dips = [], []
    up_start, up_end = np.log10(2), np.log10(1.5)
    for y in series_list:
        base = pd.Series(y).rolling(BASE_WINDOW).median().shift(1).to_numpy()
        i = BASE_WINDOW
        while i < len(y):
            b = base[i]
            if y[i] > b + up_start:
                j = i
                while j < len(y) and y[j] > b + up_end:
                    j += 1
                spikes.append(j - i)
                i = j
            elif y[i] < b - up_start:
                j = i
                while j < len(y) and y[j] < b - up_end:
                    j += 1
                dips.append(j - i)
                i = j
            else:
                i += 1
    return np.array(spikes), np.array(dips)


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(Path("data/standard").glob("*_2q.csv.gz"))
    if not files:
        raise SystemExit("No data/standard/*_2q.csv.gz found. Run adapter_ibm.py first.")

    for f in files:
        machine = f.name.replace("_2q.csv.gz", "")
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df.t, utc=True, format="mixed")
        series = [np.log10(g.sort_values("t").value.to_numpy()) for _, g in df.groupby("element")]

        pers = persistence(series)
        sp, dp = durations(series)

        print(f"\n===== {machine} =====")
        print("Share of jumps still (at least half) there afterwards:")
        print(f"  {'jump size':>9} | {'rises after 1':>13} {'after 3':>8} | {'drops after 1':>13} {'after 3':>8} |  n rises / drops")
        for thr in THRESHOLDS:
            r1, nr = pers[(thr, "rise", 1)]
            r3, _ = pers[(thr, "rise", 3)]
            d1, nd = pers[(thr, "drop", 1)]
            d3, _ = pers[(thr, "drop", 3)]
            print(f"  {'>' + str(thr) + 'x':>9} | {r1:13.0%} {r3:8.0%} | {d1:13.0%} {d3:8.0%} |  {nr} / {nd}")
        for name, d in [("Spikes (>2x baseline)", sp), ("Dips (<1/2 baseline)", dp)]:
            if len(d):
                print(f"{name}: {len(d)} events, median duration {np.median(d):.0f} calibration(s), "
                      f"{(d == 1).mean():.0%} last 1, {(d <= 2).mean():.0%} last <= 2, "
                      f"{(d >= 5).mean():.0%} last >= 5")
            else:
                print(f"{name}: none found")

        # --- plot ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        ax = axes[0]
        x = np.arange(len(THRESHOLDS))
        wdt = 0.2
        bars = [("rise", 1, "tab:red", ""), ("rise", 3, "tab:red", "//"),
                ("drop", 1, "tab:blue", ""), ("drop", 3, "tab:blue", "//")]
        for k, (kind, lag, c, hatch) in enumerate(bars):
            vals = [pers[(thr, kind, lag)][0] for thr in THRESHOLDS]
            ax.bar(x + (k - 1.5) * wdt, vals, wdt, color=c, alpha=0.75, hatch=hatch, edgecolor="black",
                   label=f"{kind}s, {lag} calibration(s) later")
        ax.set_xticks(x, [f">{t}x jump" for t in THRESHOLDS])
        ax.set_ylim(0, 1)
        ax.set_ylabel("share still >= half there")
        ax.set_title("Do jumps persist?")
        ax.legend(fontsize=8)

        ax = axes[1]
        edges = np.arange(1, MAX_DUR + 2)
        for d, c, lab in [(sp, "tab:red", "spikes (>2x)"), (dp, "tab:blue", "dips (<1/2)")]:
            if len(d):
                h = np.histogram(np.minimum(d, MAX_DUR), bins=edges)[0] / len(d)
                ax.step(edges[:-1], h, where="mid", color=c, lw=2, label=f"{lab}, n={len(d)}")
        ax.set_xticks(range(1, MAX_DUR + 1), [str(i) for i in range(1, MAX_DUR)] + [f"{MAX_DUR}+"])
        ax.set_xlabel("duration (calibrations)")
        ax.set_ylabel("share of events")
        ax.set_title("How long do they last?")
        ax.legend(fontsize=8)
        fig.suptitle(f"{machine}: behaviour of 2Q error jumps")
        fig.tight_layout()
        fig.savefig(OUT / f"diagnostic_{machine}.png", dpi=130)
    print(f"\nSaved plots/diagnostic_<machine>.png")


if __name__ == "__main__":
    main()
