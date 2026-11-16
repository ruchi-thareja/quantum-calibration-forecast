# Forecasting 2Q gate errors for noise-aware compilation on IBM Heron

Noise-aware compilers place circuits on the qubits and links with the lowest error, using the vendor's calibration data. By the time a job runs, the latest calibration (lcd) can be hours old, and it is noisy. This project asks: **does a short, recency-weighted history of calibrations predict tomorrow's errors better than the latest calibration alone, and does that give better circuits?**

It builds on [Quantum circuit compilation with historical calibration data (arXiv:2407.21462)](https://arxiv.org/abs/2407.21462), which compared the latest calibration with long averages and 15-day windows on two older IBM devices over 7–14 days.

Authors: Ruchi Thareja

## Main findings

Three 156-qubit IBM Heron machines (`ibm_fez`, `ibm_kingston`, `ibm_marrakesh`), about one year of daily calibrations each (Oct 2025 – Oct 2026). All results are on a held-out test period (the second half of the year).

1. **Recency-weighted history (~1–3 weeks) beats the latest calibration.** It predicts 2Q gate errors 20–27% better (mean absolute log10 error) and ranks the chip's links better, with bootstrap significance on all three machines.
2. **It beats a long average too.** Averaging months of data misleads after level shifts; the useful memory is about 5 calibrations.
3. **lcd's only advantage disappears with queue delay.** lcd is best at picking the top-20% links only with zero delay; with 6 h or more, history-based methods win on every metric.
4. **The weighting shape matters little.** A flat 10-day window, an EWMA, and an asymmetric EWMA (follow drops faster than rises) give nearly the same compiled circuits.
5. **Closed loop, it reduces costly placements on volatile hardware.** On `ibm_marrakesh`, lcd's circuit was >2% less likely to succeed than the asymmetric EWMA's in ~20% of runs, versus ~9% the other way. On the calmer `ibm_fez` and `ibm_kingston`, the methods are about equal.
6. **Little data is needed.** Two weeks of history is enough to tune the predictor, and settings transfer between machines.

Supporting facts from the data: error spikes usually last one calibration; about 70% of rises revert while about 60% of drops persist; a link in today's top 20% is still there 55% of the time tomorrow and 45–50% three months later.

## Repository layout

| Folder | What it does |
| --- | --- |
| `collect/` | Save your IBM account; download and flatten calibration history |
| `explore/` | Plot drift, jump persistence, good-vs-bad links, top-20% stability |
| `predict/` | Walk-forward backtests of lcd, averages, windows, EWMA and asymmetric EWMA; tuning length; queue delay |
| `closed_loop/` | Compile benchmark circuits with each predictor and score them on the next real calibration (ESP + noisy Aer simulation); bad-day analysis |
| `figures/` | Key figures from our runs |
| `logs/` | Key terminal outputs from our runs |

Calibration data is **not** included (it is large); the scripts regenerate it from your own IBM account.

## How to run

All commands run from the repository root.

```bash
pip install -r requirements.txt

# 1. Data (free IBM Open Plan account; reading calibration data uses no QPU time)
python3 collect/setup_account.py
python3 collect/log_calibration.py --backfill 365
python3 collect/log_calibration.py --compress-old

# 2. Explore
python3 explore/explore_drift.py ibm_fez
python3 predict/adapter_ibm.py            # standard per-link table used by the next steps
python3 explore/diagnostic.py
python3 explore/diagnostic_by_quality.py
python3 explore/top20_stability.py

# 3. Prediction backtests
python3 predict/backtest_ewma.py          # v1: lcd, avg, w15, EWMA
python3 predict/backtest_v2.py            # window sweep, EWMA, asymmetric EWMA, significance
python3 predict/tuning_length.py          # how much history is needed to tune
python3 predict/delay_check.py            # queue delay and quick re-measurements

# 4. Closed loop
python3 closed_loop/adapter_calibration.py
python3 closed_loop/closed_loop.py                     # 4-12 qubit circuits, ESP + noisy simulation
python3 closed_loop/closed_loop.py --circuits large    # 12-50 qubit circuits, ESP only
python3 closed_loop/tail_risk.py
```

## Methods in one line each

- **lcd**: the latest published calibration value.
- **avg**: mean of all history.
- **w-n**: mean of the last n days.
- **EWMA**: exponentially weighted mean with half-life h (in calibration cycles).
- **asym**: EWMA with a short half-life when the error drops and a longer one when it rises (h_down = 5, h_up = 10 by default).

All averaging is done on log10(error), walk-forward (each prediction uses only earlier data).

## Caveats

- The closed-loop "true device" is the next calibration, not real hardware: no crosstalk, idle decoherence or intra-day drift.
- Results cover one hardware family (IBM Heron) and small-to-medium benchmark circuits.
- Compilation uses Qiskit optimization level 3, which uses the error map for layout (VF2Layout / VF2PostLayout) but not for routing.

## Next steps

Test whether neighbouring links move together (shared qubits, crosstalk, TLS defects), and whether a qubit's T1/T2 change predicts its links' errors, before building a neighbour-aware forecast.
