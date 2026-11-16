"""
Does lcd's top-20% advantage survive queue time? Is it caused by quick re-measurements?

Run (after adapter_ibm.py):  python3 delay_check.py

Part 1 - queue delay
  Normal backtest: predict a calibration using everything measured before it.
  Here: use only data measured at least D hours BEFORE it (D = 0, 6, 12, 24, 48 h),
  as if the job was compiled D hours before it ran.
Part 2 - quick re-measurements (D = 0)
  Split predictions by how long since the link's previous measurement:
  < 12 h, 12-36 h, > 36 h. If lcd wins only on < 12 h, its edge is a sampling effect.

Methods: lcd, w10 (10-day window), asym EWMA with h_down = 5, h_up = 10
(the setting tuned on fez, which transferred well). Whole year is scored: no tuning here.
Output: printed tables + plots/delay_check.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DELAYS_H = [0, 6, 12, 24, 48]
H_DOWN, H_UP, WINDOW_DAYS = 5.0, 10.0, 10
GAP_BINS = [(0, 12, "< 12 h"), (12, 36, "12-36 h"), (36, 1e9, "> 36 h")]
METHODS = ["lcd", "w10", "asym"]
MIN_LINKS = 20
TOP_FRAC = 0.2


def predictions(df, cycle):
    """For each measurement i and delay D: predictions from data measured <= t_i - D."""
    out = []
    for el, g in df.groupby("element"):
        g = g.sort_values("t")
        if len(g) < 30:
            continue
        y = np.log10(g.value.to_numpy())
        td = (g.t - g.t.iloc[0]).dt.total_seconds().to_numpy() / 86400
        n = len(y)
        state = np.empty(n)                    # asym estimate AFTER measurement j
        m = y[0]
        state[0] = m
        for k in range(1, n):
            dc = (td[k] - td[k - 1]) / cycle
            lam = 0.5 ** (dc / (H_DOWN if y[k] < m else H_UP))
            m = lam * m + (1 - lam) * y[k]
            state[k] = m
        cs = np.concatenate([[0.0], np.cumsum(y)])
        i = np.arange(1, n)
        gap_h = (td[i] - td[i - 1]) * 24
        for D in DELAYS_H:
            j = np.searchsorted(td, td[i] - D / 24, side="right") - 1   # last index measured <= t_i - D
            j = np.minimum(j, i - 1)
            ok = j >= 0
            jj = j[ok]
            j0 = np.searchsorted(td, td[jj] - WINDOW_DAYS, side="left")
            w = (cs[jj + 1] - cs[j0]) / (jj + 1 - j0)
            out.append(pd.DataFrame({"element": el, "t": g.t.iloc[i[ok]].to_numpy(), "delay": D,
                                     "gap_h": gap_h[ok], "y": y[i[ok]],
                                     "lcd": y[jj], "w10": w, "asym": state[jj]}))
    return pd.concat(out, ignore_index=True)


def scores(p):
    """MAE over predictions, Spearman and top-20% overlap averaged over days."""
    res = {f"mae_{c}": (p[c] - p.y).abs().mean() for c in METHODS}
    sp = {c: [] for c in METHODS}
    top = {c: [] for c in METHODS}
    for _, d in p.groupby(p.t.dt.floor("D")):
        if len(d) < MIN_LINKS:
            continue
        k = max(1, int(TOP_FRAC * len(d)))
        truth = set(d.element.values[np.argsort(d.y.values, kind="stable")[:k]])
        for c in METHODS:
            sp[c].append(np.corrcoef(d[c].rank(), d.y.rank())[0, 1])
            top[c].append(len(set(d.element.values[np.argsort(d[c].values, kind="stable")[:k]]) & truth) / k)
    for c in METHODS:
        res[f"spearman_{c}"] = np.nanmean(sp[c]) if sp[c] else np.nan
        res[f"top20_{c}"] = np.nanmean(top[c]) if top[c] else np.nan
    res["n"] = len(p)
    return res


def show(rows, label):
    t = pd.DataFrame(rows).set_index(label)
    cols = [f"{k}_{c}" for k in ("mae", "spearman", "top20") for c in METHODS] + ["n"]
    print(t[cols].round(3).to_string())


def main():
    Path("plots").mkdir(exist_ok=True)
    files = sorted(Path("data/standard").glob("*_2q.csv.gz"))
    if not files:
        raise SystemExit("No data/standard/*_2q.csv.gz found. Run adapter_ibm.py first.")
    fig, axes = plt.subplots(1, len(files), figsize=(4.6 * len(files), 4), squeeze=False)
    for ax, f in zip(axes[0], files):
        m = f.name.replace("_2q.csv.gz", "")
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df.t, utc=True, format="mixed")
        cycle = float(np.nanmedian(df.sort_values("t").groupby("element").t.diff().dt.total_seconds() / 86400))
        p = predictions(df, cycle)

        print(f"\n===== {m} =====")
        print("Part 1: compiling D hours before the job runs")
        rows = []
        for D in DELAYS_H:
            r = scores(p[p.delay == D])
            r["delay_h"] = D
            rows.append(r)
        show(rows, "delay_h")
        t = pd.DataFrame(rows).set_index("delay_h")
        for c, col in zip(METHODS, ["tab:red", "tab:green", "tab:purple"]):
            ax.plot(DELAYS_H, t[f"top20_{c}"], "o-", color=col, label=c)
        ax.set_xlabel("queue delay (hours)")
        ax.set_ylabel("top-20% overlap (higher better)")
        ax.set_title(m)
        ax.legend(fontsize=8)

        print("Part 2: no delay, split by time since the link's previous measurement")
        p0 = p[p.delay == 0]
        rows = []
        for lo, hi, lab in GAP_BINS:
            part = p0[(p0.gap_h >= lo) & (p0.gap_h < hi)]
            r = scores(part)
            r["gap"] = lab
            r["share"] = len(part) / len(p0)
            rows.append(r)
        show(rows, "gap")
        print("  share of predictions per gap: " +
              ", ".join(f"{r['gap']} {r['share']:.0%}" for r in rows))
        print("  (top-20% / Spearman need >= 20 links on a day within a group; NaN = too few)")
    fig.suptitle("Does lcd's top-20% advantage survive queue time?")
    fig.tight_layout()
    fig.savefig("plots/delay_check.png", dpi=130)
    print("\nSaved plots/delay_check.png")


if __name__ == "__main__":
    main()
