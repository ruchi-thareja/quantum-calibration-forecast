"""
How stable is "being a top-20% link"?

Run (after adapter_ibm.py):  python3 top20_stability.py

Each day, the links measured that day are ranked by 2Q error; the best 20% are "top".
  1. Persistence: if a link is top today, how often is it still top k days later?
     (chance level = 20%; 100% = the top set never changes)
  2. Share of days each link is top: if a few links are top almost always and most links
     never are, "good" is mostly a fixed chip property. If many links are top sometimes,
     the best set churns.
  3. Core set: links that are top on > 80% of their days, and how much of each day's top
     set they fill.
Output: plots/top20_stability.png + printed summary.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP_FRAC = 0.2
MIN_LINKS = 20
LAGS = [1, 2, 3, 5, 7, 14, 30, 60, 90]
CORE = 0.8


def top_panel(df):
    df = df.copy()
    df["day"] = df.t.dt.floor("D")
    v = df.sort_values("t").groupby(["day", "element"]).value.last().unstack()   # day x link
    v = v[v.notna().sum(axis=1) >= MIN_LINKS]
    k = (v.notna().sum(axis=1) * TOP_FRAC).astype(int).clip(lower=1)
    ranks = v.rank(axis=1, method="first")
    top = ranks.le(k, axis=0) & v.notna()
    return top.where(v.notna())          # True / False / NaN (not measured)


def main():
    Path("plots").mkdir(exist_ok=True)
    files = sorted(Path("data/standard").glob("*_2q.csv.gz"))
    if not files:
        raise SystemExit("No data/standard/*_2q.csv.gz found. Run adapter_ibm.py first.")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for f in files:
        m = f.name.replace("_2q.csv.gz", "")
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df.t, utc=True, format="mixed")
        top = top_panel(df)
        full = top.reindex(pd.date_range(top.index.min(), top.index.max(), freq="D"))

        pers = []
        for lag in LAGS:
            a, b = full.iloc[:-lag].to_numpy(), full.iloc[lag:].to_numpy()
            ok = (a == True) & ~pd.isna(b)                      # noqa: E712
            pers.append(np.mean(b[ok] == True) if ok.any() else np.nan)

        share = top.mean()                                       # per link, over days measured
        core = share[share > CORE].index
        fill = top[core].sum(axis=1) / top.sum(axis=1)
        never = (share == 0).mean()
        rarely = (share < 0.1).mean()

        print(f"\n===== {m} =====  {top.shape[1]} links, {top.shape[0]} days")
        print("  still top-20% after k days: " +
              ", ".join(f"{lag}d {p:.0%}" for lag, p in zip(LAGS, pers)) + "   (chance = 20%)")
        print(f"  links top on > {CORE:.0%} of days (core): {len(core)}  "
              f"-> they fill {fill.mean():.0%} of a typical day's top-20% set")
        print(f"  links never top: {never:.0%}, top on < 10% of days: {rarely:.0%}")
        print(f"  links top on 10-80% of days (sometimes good): {((share >= 0.1) & (share <= CORE)).mean():.0%}")

        axes[0].plot(LAGS, pers, "o-", label=m)
        axes[1].hist(share, bins=np.linspace(0, 1, 21), histtype="step", lw=2, label=m)

    axes[0].axhline(TOP_FRAC, color="gray", ls=":", label="chance (20%)")
    axes[0].set_xscale("log")
    axes[0].set_xticks(LAGS, [str(x) for x in LAGS])
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("days later")
    axes[0].set_ylabel("share still in top 20%")
    axes[0].set_title("If a link is top today, is it top later?")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("share of days the link is in the top 20%")
    axes[1].set_ylabel("number of links")
    axes[1].set_title("Always good, never good, or sometimes?")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("plots/top20_stability.png", dpi=130)
    print("\nSaved plots/top20_stability.png")


if __name__ == "__main__":
    main()
