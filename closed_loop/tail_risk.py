"""
Bad-day analysis of the closed-loop results: how often does each method make a costly mistake?

Run (after closed_loop.py):  python3 tail_risk.py
Reads every results_cl/esp_<machine>*.csv (small and large circuit runs).

For every run (day x delay x circuit), each method's ESP is compared with the best of the three
methods on that same run ("regret" = how much worse than the best choice available):
  mean regret       average loss vs the best method
  P(loss > 2%)      share of runs where the method is more than 2% worse than the best
  P(loss > 5%)      same, more than 5%  (the "bad day" rate)
  worst 5%          the 5th percentile of the regret (typical bad case)
Head to head: share of runs where asym is > 2% worse than lcd, and vice versa, with a 95% CI
(bootstrap over days).
Output: printed tables + results_cl/tail_risk.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = ["lcd", "w10", "asym"]
IN = Path("results_cl")


def boot_share(flags, days, n=2000, seed=0):
    """95% CI of a share, resampling whole days."""
    by_day = pd.Series(flags).groupby(np.asarray(days)).agg(["sum", "count"])
    s, c = by_day["sum"].to_numpy(), by_day["count"].to_numpy()
    idx = np.random.default_rng(seed).integers(0, len(s), size=(n, len(s)))
    shares = s[idx].sum(axis=1) / c[idx].sum(axis=1)
    return np.percentile(shares, 2.5), np.percentile(shares, 97.5)


def main():
    files = sorted(IN.glob("esp_*.csv"))
    if not files:
        raise SystemExit("No results_cl/esp_*.csv found. Run closed_loop.py first.")
    plot = []
    for f in files:
        name = f.stem.replace("esp_", "")
        e = pd.read_csv(f)
        pv = e.pivot_table(index=["day", "delay", "circuit"], columns="method", values="esp").dropna()
        best = pv[METHODS].max(axis=1)
        reg = pv[METHODS].div(best, axis=0) - 1                       # <= 0, 0 = best choice
        print(f"\n===== {name} =====  {len(pv)} runs, circuits: {', '.join(sorted(e.circuit.unique()))}")
        rows = []
        for D, g in reg.groupby(level="delay"):
            for m in METHODS:
                r = g[m]
                rows.append({"delay_h": D, "method": m, "mean_regret_%": 100 * r.mean(),
                             "P(loss>2%)": (r < -0.02).mean(), "P(loss>5%)": (r < -0.05).mean(),
                             "worst5%_%": 100 * r.quantile(0.05)})
                plot.append({"file": name, "delay": D, "method": m,
                             "p2": (r < -0.02).mean(), "p5": (r < -0.05).mean()})
        print(pd.DataFrame(rows).round(3).to_string(index=False))

        print("  head to head (share of runs more than 2% worse than the other, 95% CI over days):")
        for D, g in pv.groupby(level="delay"):
            days = g.index.get_level_values("day")
            a_bad = (g["asym"] / g["lcd"] - 1) < -0.02
            l_bad = (g["lcd"] / g["asym"] - 1) < -0.02
            lo1, hi1 = boot_share(a_bad.to_numpy(), days)
            lo2, hi2 = boot_share(l_bad.to_numpy(), days)
            print(f"    delay {D:>4g} h: asym worse than lcd {a_bad.mean():.1%} [{lo1:.1%}, {hi1:.1%}]   "
                  f"lcd worse than asym {l_bad.mean():.1%} [{lo2:.1%}, {hi2:.1%}]")

    p = pd.DataFrame(plot)
    files_ = list(p.file.unique())
    fig, axes = plt.subplots(1, len(files_), figsize=(4.4 * len(files_), 3.8), squeeze=False)
    colors = {"lcd": "tab:red", "w10": "tab:green", "asym": "tab:purple"}
    for ax, fn in zip(axes[0], files_):
        q = p[p.file == fn]
        delays = sorted(q.delay.unique())
        x = np.arange(len(delays))
        w = 0.8 / len(METHODS)
        for k, m in enumerate(METHODS):
            v = [q[(q.delay == D) & (q.method == m)].p5.iloc[0] for D in delays]
            ax.bar(x + (k - 1) * w, v, w, color=colors[m], label=m)
        ax.set_xticks(x, [f"{D:g} h" for D in delays])
        ax.set_xlabel("queue delay")
        ax.set_ylabel("share of runs > 5% worse than the best method")
        ax.set_title(fn, fontsize=9)
        ax.legend(fontsize=8)
    fig.suptitle("Bad-day rate: how often a method loses more than 5% (lower is better)", fontsize=10)
    fig.tight_layout()
    fig.savefig(IN / "tail_risk.png", dpi=130)
    print(f"\nSaved {IN}/tail_risk.png")


if __name__ == "__main__":
    main()
