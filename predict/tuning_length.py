"""
How much history is needed to TUNE the predictor? Is re-tuning every week better?

Run (after backtest_v2.py, it reads results_v2/daily_<machine>.csv):
    python3 tuning_length.py

The predictions themselves never change (they always use only the recent past).
What changes here is how h_down / h_up (and the window length) are CHOSEN:

  fixed-L     choose once, using only the L weeks just before the test period
              (L = 1, 2, 4, 8, 12 weeks, or the full first half as before)
  rolling-L   every week of the test period, re-choose using the previous L weeks
  oracle      the best choice in hindsight on the test period itself
              (not achievable, it only shows the ceiling)

Test period = second half, exactly as in backtest_v2.py.
Outputs: printed tables + results_v2/tuning_length.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TUNE_METRIC = "mae"          # same choice as backtest_v2.py
LENGTHS_WEEKS = [1, 2, 4, 8, 12]
ROLLING_WEEKS = [2, 4, 8]
FAMILIES = {"asym": "asym_", "window": "w"}
IN = Path("results_v2")


def load(path):
    d = pd.read_csv(path, header=[0, 1], index_col=0)
    d.index = pd.to_datetime(d.index, utc=True, format="mixed")
    return {m: d[m] for m in ("mae", "spearman", "top20")}


def best_col(dm, cols, mask):
    s = dm[TUNE_METRIC].loc[mask, cols].mean()
    return s.idxmin() if TUNE_METRIC == "mae" else s.idxmax()


def boot(a, b, n=2000, seed=0):
    d = (a - b).dropna().to_numpy()
    bs = np.random.default_rng(seed).choice(d, size=(n, len(d))).mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    files = sorted(IN.glob("daily_*.csv"))
    if not files:
        raise SystemExit("No results_v2/daily_*.csv found. Run backtest_v2.py first.")
    plot_rows = {}
    for f in files:
        machine = f.stem.replace("daily_", "")
        dm = load(f)
        days = dm["mae"].index
        split = days.min() + (days.max() - days.min()) / 2
        test = days >= split
        all_cols = list(dm["mae"].columns)
        print(f"\n===== {machine} =====  test period starts {split:%Y-%m-%d}")

        rows = []
        for fam, prefix in FAMILIES.items():
            cols = [c for c in all_cols if c.startswith(prefix) and (fam != "window" or c[1:].isdigit())]

            def score(daily_choice, label):
                # daily_choice: Series day -> chosen column (only test days)
                vals = {m: np.array([dm[m].at[d, c] for d, c in daily_choice.items()]) for m in dm}
                rows.append({"family": fam, "tuning": label, "chosen": daily_choice.mode().iloc[0]
                             if daily_choice.nunique() > 1 else daily_choice.iloc[0],
                             **{m: np.nanmean(v) for m, v in vals.items()}})
                return pd.Series(vals[TUNE_METRIC], index=daily_choice.index)

            test_days = days[test]
            series = {}
            c = best_col(dm, cols, ~test)
            series["fixed, full first half"] = score(pd.Series(c, index=test_days), "fixed, full first half")
            for L in LENGTHS_WEEKS:
                mask = (days < split) & (days >= split - pd.Timedelta(weeks=L))
                c = best_col(dm, cols, mask)
                series[f"fixed, last {L} wk"] = score(pd.Series(c, index=test_days), f"fixed, last {L} wk")
            for L in ROLLING_WEEKS:
                choice = {}
                for wk_start in pd.date_range(split, days.max(), freq="7D"):
                    mask = (days < wk_start) & (days >= wk_start - pd.Timedelta(weeks=L))
                    c = best_col(dm, cols, mask)
                    for d in test_days[(test_days >= wk_start) & (test_days < wk_start + pd.Timedelta(days=7))]:
                        choice[d] = c
                series[f"rolling, last {L} wk"] = score(pd.Series(choice), f"rolling, last {L} wk")
            c = best_col(dm, cols, test)
            series["oracle (hindsight)"] = score(pd.Series(c, index=test_days), "oracle (hindsight)")

            if fam == "asym":
                ref = series["fixed, full first half"]
                print(f"  significance vs 'fixed, full first half' ({TUNE_METRIC}, 95% CI of difference):")
                for k in ["fixed, last 2 wk", "fixed, last 4 wk", "rolling, last 4 wk"]:
                    d, lo, hi = boot(series[k], ref)
                    better = (hi < 0) if TUNE_METRIC == "mae" else (lo > 0)
                    worse = (lo > 0) if TUNE_METRIC == "mae" else (hi < 0)
                    v = "better" if better else ("worse" if worse else "no significant difference")
                    print(f"    {k:22s}: {d:+.4f} [{lo:+.4f}, {hi:+.4f}] -> {v}")

        lcd = {m: dm[m].loc[test, "lcd"].mean() for m in dm}
        rows.append({"family": "lcd", "tuning": "-", "chosen": "lcd", **lcd})
        tab = pd.DataFrame(rows)
        print(tab.round(4).to_string(index=False))
        plot_rows[machine] = (tab, lcd)

    # --- plot: test MAE and top-20 vs tuning length, asym family ---
    machines = list(plot_rows)
    fig, axes = plt.subplots(2, len(machines), figsize=(4.6 * len(machines), 7), squeeze=False)
    for j, m in enumerate(machines):
        tab, lcd = plot_rows[m]
        a = tab[tab.family == "asym"].set_index("tuning")
        for i, metric in enumerate(["mae", "top20"]):
            ax = axes[i, j]
            xs = LENGTHS_WEEKS
            ax.plot(xs, [a.loc[f"fixed, last {L} wk", metric] for L in xs], "o-", label="fixed, last L weeks")
            ax.plot(ROLLING_WEEKS, [a.loc[f"rolling, last {L} wk", metric] for L in ROLLING_WEEKS], "s-",
                    label="rolling, re-tuned weekly")
            ax.axhline(a.loc["fixed, full first half", metric], color="gray", ls="--", label="fixed, full half year")
            ax.axhline(a.loc["oracle (hindsight)", metric], color="tab:green", ls=":", label="oracle (ceiling)")
            ax.axhline(lcd[metric], color="tab:red", ls="--", label="lcd")
            ax.set_xscale("log")
            ax.set_xticks(xs, [str(x) for x in xs])
            ax.set_xlabel("tuning history L (weeks)")
            ax.set_ylabel(("mean |log10 error| (lower better)" if metric == "mae"
                           else "top-20% overlap (higher better)"), fontsize=8)
            if i == 0:
                ax.set_title(f"{m}: asymmetric EWMA")
            ax.legend(fontsize=7)
    fig.suptitle("Test period: how much history is needed to tune h_down / h_up?")
    fig.tight_layout()
    fig.savefig(IN / "tuning_length.png", dpi=130)
    print(f"\nSaved {IN}/tuning_length.png")


if __name__ == "__main__":
    main()
