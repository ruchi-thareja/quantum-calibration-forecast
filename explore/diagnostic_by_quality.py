"""
Diagnostic split by link quality: do GOOD links behave differently from the rest?

Run (after adapter_ibm.py):  python3 diagnostic_by_quality.py

Hypothesis being tested
  lcd finds the best links better than history methods. Possible reason: for good links,
  changes are mostly real recalibration effects that persist, while for bad links they are
  mostly spikes. If true, good links should show
    - more persistence of jumps (rises AND drops), and
    - higher "memory": a deviation from baseline today predicts a deviation tomorrow.

Definitions (only past data used for the baseline)
  baseline  = median of the previous 7 calibrations of that link
  good      = baseline in the best 20% of all baselines on that machine
  persisted = at least half of a jump is still there 1 / 3 calibrations later
  memory    = correlation between today's and the next deviation from baseline
              (0 = deviations are pure one-off noise, 1 = they fully persist)
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE_WINDOW = 7
GOOD_FRAC = 0.2
THRESHOLDS = [1.2, 1.5, 2]


def main():
    files = sorted(Path("data/standard").glob("*_2q.csv.gz"))
    if not files:
        raise SystemExit("No data/standard/*_2q.csv.gz found. Run adapter_ibm.py first.")
    for f in files:
        machine = f.name.replace("_2q.csv.gz", "")
        df = pd.read_csv(f)
        df["t"] = pd.to_datetime(df.t, utc=True, format="mixed")
        series = []
        for _, g in df.groupby("element"):
            y = np.log10(g.sort_values("t").value.to_numpy())
            if len(y) < BASE_WINDOW + 5:
                continue
            base = pd.Series(y).rolling(BASE_WINDOW).median().shift(1).to_numpy()
            series.append((y, base))
        q = np.nanquantile(np.concatenate([b for _, b in series]), GOOD_FRAC)

        rec = {k: [] for k in ["good", "jump", "k1", "k3", "dev0", "dev1"]}
        for y, base in series:
            i = np.arange(BASE_WINDOW, len(y) - 3)
            jump = y[i] - y[i - 1]
            safe = np.where(jump == 0, np.nan, jump)
            rec["good"].append(base[i] <= q)
            rec["jump"].append(jump)
            rec["k1"].append((y[i + 1] - y[i - 1]) / safe)
            rec["k3"].append((y[i + 3] - y[i - 1]) / safe)
            rec["dev0"].append(y[i] - base[i])
            rec["dev1"].append(y[i + 1] - base[i + 1])
        r = {k: np.concatenate(v) for k, v in rec.items()}

        print(f"\n===== {machine} =====  (good = baseline error below {10 ** q:.2e})")
        print(f"  {'group':6} {'jump':>6} | {'rises kept 1/3 later':>21} | {'drops kept 1/3 later':>21} | "
              f"{'rises/drops per 100 cal.':>24}")
        for grp_name, gmask in [("good", r["good"]), ("rest", ~r["good"])]:
            n_cal = gmask.sum()
            for thr in THRESHOLDS:
                L = np.log10(thr)
                up = gmask & (r["jump"] > L)
                dn = gmask & (r["jump"] < -L)
                print(f"  {grp_name:6} {'>' + str(thr) + 'x':>6} | "
                      f"{np.mean(r['k1'][up] >= 0.5):9.0%} / {np.mean(r['k3'][up] >= 0.5):4.0%}       | "
                      f"{np.mean(r['k1'][dn] >= 0.5):9.0%} / {np.mean(r['k3'][dn] >= 0.5):4.0%}       | "
                      f"{100 * up.sum() / n_cal:10.1f} / {100 * dn.sum() / n_cal:4.1f}")
        for grp_name, gmask in [("good", r["good"]), ("rest", ~r["good"])]:
            ok = gmask & np.isfinite(r["dev0"]) & np.isfinite(r["dev1"])
            mem = np.corrcoef(r["dev0"][ok], r["dev1"][ok])[0, 1]
            typ = np.median(np.abs(r["dev0"][ok]))
            print(f"  memory ({grp_name}): {mem:+.2f}   typical deviation from baseline: x{10 ** typ:.2f}")


if __name__ == "__main__":
    main()
