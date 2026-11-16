"""
Step 1 backtest: how well do lcd, avg, w-15 and EWMA predict the NEXT real calibration?

Run (after adapter_ibm.py):  python3 backtest_ewma.py
Works on every data/standard/*_2q.csv.gz file, i.e. any machine / vendor with an adapter.

Logic
  - Everything in log10(error): errors change multiplicatively.
  - Walk forward: each prediction uses ONLY measurements before it.
  - EWMA half-life h is in CALIBRATION CYCLES (cycle = median time between remeasurements
    on that machine), so the same h means the same thing on platforms that calibrate
    at different rates. After a gap, the old estimate automatically counts for less.
  - Time split: first half of each machine's history = tuning (pick h),
    second half = test (judge the methods; never used for tuning).

Outputs (results/ folder)
  h_curve.png           error and ranking quality vs h, per machine, with lcd/avg/w-15 lines
  scores.csv            every method x machine x split
  predictions_<m>.csv.gz all predictions, for later steps (sigma, coverage, ML)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

H_GRID = [0.5, 1, 2, 3, 5, 7, 10, 15, 30, 60]   # half-lives, in calibration cycles
WINDOW_DAYS = 15                                # the paper's w-15
MIN_POINTS = 30                                 # skip links with too little history
MIN_LINKS_PER_DAY = 20                          # for daily ranking scores
TOP_FRAC = 0.2                                  # "top 20% best links" overlap
EWMA_COLS = [f"ewma_h{h:g}" for h in H_GRID]
BASE_COLS = ["lcd", "avg", f"w{WINDOW_DAYS}"]
OUT = Path("results")


def predict_machine(df):
    """Walk-forward predictions for every real measurement of every link."""
    df = df.sort_values(["element", "t"])
    gaps = df.groupby("element").t.diff().dt.total_seconds() / 86400
    cycle = float(np.nanmedian(gaps))
    H = np.array(H_GRID)
    out = []
    for el, g in df.groupby("element"):
        if len(g) < MIN_POINTS:
            continue
        y = np.log10(g.value.to_numpy())
        td = (g.t - g.t.iloc[0]).dt.total_seconds().to_numpy() / 86400
        csum = np.concatenate([[0.0], np.cumsum(y)])
        n = len(y)
        preds = np.empty((n - 1, len(BASE_COLS) + len(H)))
        m = np.full(len(H), y[0])                      # EWMA state, one per half-life
        for i in range(1, n):
            j0 = np.searchsorted(td, td[i] - WINDOW_DAYS, side="left")
            win = csum[i] - csum[j0] if j0 < i else np.nan
            preds[i - 1, 0] = y[i - 1]                              # lcd
            preds[i - 1, 1] = csum[i] / i                           # avg (all history)
            preds[i - 1, 2] = win / (i - j0) if j0 < i else y[i - 1]  # w-15
            preds[i - 1, 3:] = m                                    # EWMA, before seeing y[i]
            lam = 0.5 ** (((td[i] - td[i - 1]) / cycle) / H)        # bigger gap -> trust old less
            m = lam * m + (1 - lam) * y[i]
        block = pd.DataFrame(preds, columns=BASE_COLS + EWMA_COLS)
        block.insert(0, "y", y[1:])
        block.insert(0, "t", g.t.iloc[1:].to_numpy())
        block.insert(0, "element", el)
        out.append(block)
    return pd.concat(out, ignore_index=True), cycle


def daily_rank_scores(p, col):
    """Per day: Spearman (pred vs actual ranking) and top-20% overlap, averaged over days."""
    sp, ov = [], []
    for _, d in p.groupby(p.t.dt.floor("D")):
        if len(d) < MIN_LINKS_PER_DAY:
            continue
        a, b = d[col].rank(), d.y.rank()
        sp.append(np.corrcoef(a, b)[0, 1])
        k = max(1, int(TOP_FRAC * len(d)))
        ov.append(len(set(d.nsmallest(k, col).element) & set(d.nsmallest(k, "y").element)) / k)
    return np.nanmean(sp), np.nanmean(ov)


def score(p, cols):
    rows = []
    for c in cols:
        err = (p[c] - p.y).abs()
        sp, ov = daily_rank_scores(p, c)
        rows.append({"method": c, "mae_log10": err.mean(), "median_ae_log10": err.median(),
                     "spearman": sp, "top20_overlap": ov, "n": len(p)})
    return pd.DataFrame(rows)


def bootstrap_vs_lcd(p, col, n_boot=2000, seed=0):
    """95% CI of (daily MAE of col) - (daily MAE of lcd). Negative = col better than lcd."""
    day = p.t.dt.floor("D")
    d = ((p[col] - p.y).abs().groupby(day).mean() - (p.lcd - p.y).abs().groupby(day).mean()).to_numpy()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(Path("data/standard").glob("*_2q.csv.gz"))
    if not files:
        raise SystemExit("No data/standard/*_2q.csv.gz found. Run adapter_ibm.py first.")

    all_scores, preds, splits = [], {}, {}
    for f in files:
        machine = f.name.replace("_2q.csv.gz", "")
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df.t, utc=True, format="mixed")
        p, cycle = predict_machine(df)
        p.to_csv(OUT / f"predictions_{machine}.csv.gz", index=False)
        t_split = p.t.min() + (p.t.max() - p.t.min()) / 2
        splits[machine] = t_split
        preds[machine] = p
        print(f"\n{machine}: {p.element.nunique()} links, {len(p)} predictions, "
              f"cycle = {cycle:.2f} days, split at {t_split:%Y-%m-%d}")
        for name, part in [("tune", p[p.t < t_split]), ("test", p[p.t >= t_split])]:
            s = score(part, BASE_COLS + EWMA_COLS)
            s.insert(0, "split", name)
            s.insert(0, "machine", machine)
            all_scores.append(s)
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(OUT / "scores.csv", index=False)

    # --- pick h on each machine's TUNING half; also take fez's h to test transfer ---
    tune = scores[(scores.split == "tune") & scores.method.str.startswith("ewma")]
    best_h = tune.loc[tune.groupby("machine").mae_log10.idxmin()].set_index("machine").method
    ref = "ibm_fez" if "ibm_fez" in best_h.index else best_h.index[0]
    print("\nBest half-life on tuning data:",
          {m: c.replace("ewma_h", "") + " cycles" for m, c in best_h.items()})

    print(f"\n=== TEST period (never used for tuning) — lower MAE / higher ranking is better ===")
    for machine, p in preds.items():
        test = p[p.t >= splits[machine]]
        cols = BASE_COLS + [best_h[machine]]
        if best_h[ref] not in cols:
            cols.append(best_h[ref])
        s = score(test, cols)
        s["method"] = s.method.replace({best_h[machine]: f"{best_h[machine]} (own h)"})
        if best_h[ref] != best_h[machine]:
            s["method"] = s.method.replace({best_h[ref]: f"{best_h[ref]} (h from {ref})"})
        print(f"\n{machine}")
        print(s.drop(columns="n").round(4).to_string(index=False))
        diff, lo, hi = bootstrap_vs_lcd(test, best_h[machine])
        verdict = "significantly better than lcd" if hi < 0 else (
            "significantly worse than lcd" if lo > 0 else "not significantly different from lcd")
        print(f"  EWMA (own h) minus lcd, daily MAE: {diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] -> {verdict}")

    # --- curves: error and ranking vs h, tuning half ---
    machines = list(preds)
    fig, axes = plt.subplots(2, len(machines), figsize=(4.5 * len(machines), 7), squeeze=False)
    styles = {"lcd": ("tab:red", "--"), "avg": ("tab:gray", ":"), f"w{WINDOW_DAYS}": ("tab:green", "-.")}
    for j, machine in enumerate(machines):
        s = scores[(scores.machine == machine) & (scores.split == "tune")].set_index("method")
        for i, metric in enumerate(["mae_log10", "spearman"]):
            ax = axes[i, j]
            ax.plot(H_GRID, s.loc[EWMA_COLS, metric], "o-", color="tab:blue", label="EWMA")
            for b, (c, ls) in styles.items():
                ax.axhline(s.loc[b, metric], color=c, ls=ls, label=b)
            ax.set_xscale("log")
            ax.set_xlabel("EWMA half-life (calibration cycles)")
            ax.set_ylabel("mean |log10 error| (lower = better)" if i == 0 else "daily rank correlation (higher = better)")
            if i == 0:
                ax.set_title(machine)
            ax.legend(fontsize=8)
    fig.suptitle("Tuning period: prediction of the next calibration vs EWMA half-life")
    fig.tight_layout()
    fig.savefig(OUT / "h_curve.png", dpi=130)
    print(f"\nSaved {OUT}/h_curve.png, {OUT}/scores.csv and predictions files.")


if __name__ == "__main__":
    main()
