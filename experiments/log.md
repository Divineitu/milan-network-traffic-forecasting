# Experiment Log

Every training run gets one entry. Validation = Dec 9–15; the test week (Dec 16–22) is only used for the final runs.


## Phase 1 — data handling and memory (single daily file, 308 MB of text)

Measured with `src/data/memory_profile.py`. Each variant runs in a fresh subprocess, because memory
freed by Python is not returned to the OS. "Data memory" excludes the ~109 MB interpreter + pandas baseline.

| # | Variant | Data peak | Table in RAM | Time | Reasoning → next change |
|---|---------|-----------|--------------|------|-------------------------|
| 1 | naive `read_csv` (8 cols, 64-bit) | 591.9 MB | 295.6 MB | 6.9 s | 17.9 GB extrapolated over 62 days > 16 GB RAM: impossible on this machine. Drop unused columns, shrink dtypes. |
| 2 | 3 cols, int16/float32, groupby | 255.4 MB | 19.2 MB | 6.0 s | Table shrank 15.4x but peak only 1.9x: `read_csv` must parse all 308 MB before any column is dropped. Parse in chunks. |
| 3 | + chunked reading (500k rows) | **130.0 MB** | 19.2 MB | 6.7 s | 4.6x lower peak than naive, ~no time cost. Peak now depends on chunk size, not file size. Adopted. |

## Phase 1 — full pipeline over 62 files (`src/data/preprocess.py`)

| # | Change | Peak process memory | Reasoning → next change |
|---|--------|--------------------|-------------------------|
| 4 | First full run | 1,271 MB | Per-file processing only used ~531 MB, so something else dominated. Profile the stages. |
| 5 | Block-wise per-square summary (500 squares at a time) | 1,271 MB (unchanged) | Isolated test proved the fix works (827 → 422 MB on the saved matrix), but it was masked by a larger effect. Keep it, and keep profiling. |
| 6 | Stage profiling | — | Per-file stages peak at only 239 MB; the peak *grew ~11 MB per file* (581 → 656 MB over 4 files), extrapolating to ~1,260 MB over 62 files. An accumulation across iterations, not one big allocation. |
| 7 | `del` intermediates + `gc.collect()` per file | ~582 MB, flat across files | Confirmed on a 6-file test: RSS stays at 453 MB instead of climbing. Adopted. |

**Outputs:** `data/processed/traffic_matrix.npy` (8,928 x 10,000 float32, 341 MB, memory-mappable),
`time_index.parquet`, `square_totals.parquet`. 19.4 GB of text → 341 MB (57x), 0.04% of slots missing.

## Phase 2 — naive baselines on the validation week (Dec 9–15), no parameters fitted

Reference point for every model: a model is only useful if it beats these.

| Area | Baseline | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 5161 | persistence x̂(t+1)=x(t) | 116.8 | 175.2 | 9.5% |
| 5161 | same time yesterday (lag 144) | 294.9 | 525.5 | 18.3% |
| 5161 | same time last week (lag 1008) | 214.3 | 320.1 | 15.7% |
| 5059 | persistence | 99.9 | 145.8 | 8.3% |
| 5059 | lag 144 | 213.4 | 308.0 | 17.8% |
| 5059 | lag 1008 | 203.4 | 300.4 | 13.9% |
| 5259 | persistence | 89.9 | 132.2 | 7.8% |
| 5259 | lag 144 | 513.1 | 898.8 | 66.8% |
| 5259 | lag 1008 | 184.8 | 295.7 | 14.5% |

Observations: persistence is by far the strongest naive method at 10-minute resolution (lag-1 ACF = 0.987).
Lag 1008 beats lag 144 everywhere, and lag 144 fails badly in Brera (5259) because weekday/weekend
alternation makes "yesterday" the wrong kind of day. Both findings argue for giving models the recent
steps *and* weekly-lag information, rather than a daily lag alone.

## Phase 4 — hyperparameter experiments (square 5161, scored on validation week Dec 9–15)

Full runs in `experiments/results/tuning_arima.csv` and `tuning_lgbm.csv`. Reference: persistence MAE 116.83.

### ARIMA + Fourier (staged search, 17 runs)

| Stage | What was varied | Result | Reasoning → next step |
|---|---|---|---|
| 1 | Fourier harmonics (day 3/4/6 × week 1/2), ARIMA fixed at (2,0,0) | best (6,1), MAE 103.38; spread only 103.4–104.5 | More daily detail helps slightly; the weekly term adds nothing here (w1 ≈ w2), because the validation week contains no holiday. Fix (6,1) and tune the ARIMA order. |
| 2 | orders (1,0,0) … (2,1,1) | (3,0,0) MAE **101.26**; (1,0,0) 111.56 | AR(1) is clearly insufficient, confirming the PACF (significant at lags 1–2, with lag 3 still contributing). Improvement continued to the edge of the grid, so extend it. |
| 3 | orders (4,0,0), (5,0,0), (6,0,0), (8,0,0) | 101.82 / 102.00 / 102.02 / 102.02 | Error rises again: p=3 is a true optimum, not a grid boundary. **Selected ARIMA(3,0,0)+Fourier(d6,w1)**, validation MAE 101.26, RMSE 150.86, fit 8.2 s. |

### LightGBM (24 runs)

| Round | What was varied | Result | Reasoning → next step |
|---|---|---|---|
| 1 | lr {0.03, 0.05, 0.1} × leaves {15, 31, 63} × trees {200, 400} | best leaves 15 / lr 0.03 / 200 trees, MAE 99.27; worst leaves 63 / lr 0.1 / 400 trees, MAE 116.64 | Error rises monotonically with capacity and with the learning rate: the model is overfitting ~5.5k training rows. Search *below* the best point instead of around it. |
| 2 | leaves {7, 15} × trees {100, 200, 300}, lr 0.03, min_child_samples 40, feature_fraction 0.7 | **leaves 15, 300 trees, MAE 98.49**, RMSE 154.36, fit 0.29 s | Stronger regularisation let more trees help rather than hurt. leaves 7 was worse (101.28), so capacity is now correctly balanced. **Selected.** |

**Cross-model note (validation):** ARIMA has the better RMSE (150.86 vs 154.36) while LightGBM has the
better MAE (98.49 vs 101.26) and MAPE (7.54% vs 8.55%). So LightGBM is better on typical errors and ARIMA
on the largest ones — a trade-off worth testing on the test week rather than declaring a single winner.

### LSTM — diagnosis before tuning (square 5161)

| # | Run | Result | Reasoning → next change |
|---|-----|--------|-------------------------|
| D1 | hidden 32, level target, batch 128, max 25 epochs, patience 4 | MAE **214.63** (persistence 116.83), fit 15.8 s | Far worse than persistence, and training stopped after only a few epochs. Is it undertrained or mis-specified? Train without early stopping and inspect the curves. |
| D2 | hidden 64, level target, 40 epochs, **no early stopping** | train loss falls smoothly to ~epoch 35; internal validation loss is **noisy** (e.g. 0.0017 → 0.0031 → 0.0016 between epochs); final MAE **123.40**, still worse than persistence; 3.6 s/epoch | (a) Patience 4 stopped training on single noisy epochs → raise patience to 10, cap 60 epochs. (b) Even fully trained, predicting the **level** loses to persistence: with lag-1 ACF 0.987 the network must reproduce its last input almost exactly before adding anything, and small relative errors in the level are large absolute errors at peak traffic. → Add a **delta target** (predict x(t+1)−x(t), standardised, added back to x(t)), so persistence is the starting point and the network learns only the correction. Test level vs delta explicitly as stage 1 of tuning. |

Note: an earlier estimate of ~16 s/epoch came from a run under severe memory pressure (0.6 GB free);
with memory freed the measured cost is 3.6 s/epoch.

### LSTM — staged tuning (8 runs; `experiments/results/tuning_lstm.csv`)

Batch 128, up to 60 epochs, early-stopping patience 10. Reference: persistence MAE 116.83.
The 9th planned run (48-hour window) was interrupted by the OS low-memory guard and is not reported.

| Stage | Change from base (delta, hidden 64, 1 layer, lr 1e-3, window 144) | MAE | RMSE | MAPE | Fit | Reasoning |
|---|---|---|---|---|---|---|
| 1 | **level** target | 127.41 | 189.33 | 11.18% | 142 s | Worse than persistence, confirming diagnostic D2. |
| 1 | **delta** target | 103.27 | 153.89 | 9.03% | 112 s | **−24.1 MAE vs level**: the single largest improvement in any tuning run. Adopted for stage 2. |
| 2 | hidden 32 | 103.66 | 154.90 | 8.85% | 75 s | Barely worse than 64 at 2/3 the cost: capacity is not the bottleneck. |
| 2 | hidden 128 | 103.06 | 155.61 | 8.56% | 203 s | No meaningful gain for twice the cost. Width is saturated. |
| 2 | 2 layers + dropout 0.2 | 99.21 | **148.52** | 8.80% | 824 s | Best RMSE (depth helps on the large errors) but **7× slower**. |
| 2 | **lr 3e-3** | **98.32** | 149.05 | **7.98%** | 224 s | Best MAE and MAPE: the default step size was too small for the epoch budget. **Selected.** |
| 2 | lr 3e-4 | 104.99 | 159.59 | 8.56% | 250 s | Slower learning is worse, consistent with the 3e-3 result. |
| 2 | window 36 (6 h) | 104.79 | 159.05 | 8.38% | 66 s | Worse than 144: although the PACF is concentrated at lags 1–2, the network still benefits from seeing a full daily cycle. |

**Selected LSTM:** delta target, hidden 64, 1 layer, lr 3e-3, window 144 — validation MAE 98.32, RMSE 149.05, MAPE 7.98%.
The 2-layer model has marginally better RMSE (148.52) but costs 824 s vs 224 s per fit, a poor trade on a CPU.

### Validation summary across the three models (square 5161)

| Model | MAE | RMSE | MAPE | Fit time |
|---|---|---|---|---|
| Persistence | 116.83 | 175.19 | 9.46% | 0 s |
| ARIMA(3,0,0)+Fourier(d6,w1) | 101.26 | 150.86 | 8.55% | 8.2 s |
| LightGBM (leaves 15, lr 0.03, 300 trees) | 98.49 | 154.36 | 7.54% | 0.29 s |
| LSTM (delta, hidden 64, lr 3e-3) | **98.32** | **149.05** | 7.98% | 224 s |

The three tuned models are within 3% of each other on MAE, and all beat persistence by 13–16%.
The LSTM is marginally best on MAE and RMSE, LightGBM on MAPE, and LightGBM is ~770× faster to train
than the LSTM. Whether these small validation differences survive on the test week, and are statistically
significant, is tested in Phase 5.

## Phase 4 — FINAL test-week results (Dec 16–22), run once with the validation-selected configurations

Models refitted on train+validation (Nov 1 – Dec 15). One-step-ahead from the true history.
Raw outputs: `experiments/results/final_metrics.csv`, `final_timing.csv`, `predictions_<square>.csv`.

| Area | Model | MAE | RMSE | MAPE |
|---|---|---|---|---|
| 5161 (Duomo) | Persistence | 92.80 | 134.88 | 9.19% |
| | Seasonal naive (week) | 300.75 | 432.95 | 28.97% |
| | ARIMA+Fourier | 80.34 | 119.57 | 8.72% |
| | LightGBM | 89.12 | 130.36 | 9.33% |
| | **LSTM** | **79.26** | **118.28** | **8.38%** |
| 5059 (Duomo) | Persistence | 81.52 | 114.38 | 7.96% |
| | Seasonal naive (week) | 259.95 | 361.12 | 25.73% |
| | **ARIMA+Fourier** | **68.07** | **95.13** | 7.45% |
| | LightGBM | 77.50 | 108.80 | 8.00% |
| | LSTM | 69.70 | 98.08 | **6.96%** |
| 5259 (Brera) | Persistence | 75.97 | 109.58 | 8.11% |
| | Seasonal naive (week) | 210.24 | 289.50 | 28.56% |
| | ARIMA+Fourier | 71.29 | 96.23 | 9.50% |
| | LightGBM | 65.89 | 93.77 | 7.53% |
| | **LSTM** | **61.75** | **88.96** | **6.86%** |

**Timing (mean ± std over the 3 areas; i7-8665U, CPU only):** fit — LightGBM 1.56 ± 2.26 s, ARIMA 6.17 ± 1.51 s,
LSTM 114.9 ± 30.4 s; inference for 1,008 steps — LightGBM 0.060 s, ARIMA 0.114 s, LSTM 0.233 s.
Caveat: LightGBM's first fit (4.2 s, square 5161) includes one-off library initialisation; later fits took 0.2–0.3 s.

## Phase 5 — error analysis (`src/experiments/analyse_errors.py`, fig8–fig10)

**Skill vs persistence (1 − MAE/MAE_persistence):**

| Model | 5161 | 5059 | 5259 |
|---|---|---|---|
| ARIMA+Fourier | 0.134 | **0.165** | 0.062 |
| LightGBM | 0.040 | 0.049 | 0.133 |
| LSTM | **0.146** | 0.145 | **0.187** |

The ranking changes by area: ARIMA is strong in the two Duomo squares but weak in Brera; LightGBM is the
reverse; the LSTM is consistently good everywhere.

**Diebold–Mariano (squared error, Newey–West variance, 5% level):** LSTM significantly beats LightGBM in all
three areas. LSTM vs ARIMA: *no significant difference* in 5161 (p=0.71) and 5059 (p=0.26); LSTM better in
5259 (p=0.012). ARIMA beats LightGBM in 5161 and 5059. **LightGBM is not significantly better than
persistence in either Duomo square** (p=0.37, p=0.056). All other models beat persistence (p<0.001).

**Error by hour (fig9):** errors are ~2% of the area mean at night and 8–10% between 13:00 and 18:00.
ARIMA is worst on the 08:00–09:00 morning ramp (fixed seasonal shape); LightGBM is worst at 18:00 and 21:00.

**Worst 6-hour windows:** Tue 17 Dec 13:00–18:50 in *both* Duomo squares (adjacent cells, the same
afternoon ⇒ a shared local cause), and Thu 19 Dec 12:30–18:20 in Brera. In fig10 every model tracks the
actual curve one step late during the volatile peak: high-frequency fluctuations are effectively
unpredictable one step ahead, so the models fall back to persistence-like behaviour there.

### Post-hoc diagnostic: why is LightGBM weak in the Duomo? (NOT used for model selection)

| Hypothesis | Test | Result |
|---|---|---|
| H1: trees cannot extrapolate beyond the training range | Count test values above the train+val maximum | **Rejected**: 0 of 1,008 in every area (test max 5,496 vs train max 8,044 in 5161) |
| H2: reliance on the weekly lag misleads it when the week-to-week level shifts | Feature importance; ablation without lag_1008 | **Supported**: lag_1008 carries 9.9% (5161) and 7.3% (5059) of gain but is not in the top features for 5259 (where LightGBM does well). Removing it on the **test** week improves MAE in all areas: 5161 89.12 → 83.01, 5059 77.50 → 74.31, 5259 65.89 → 63.89 |

Weekly mean traffic: Dec 2–8 → Dec 9–15 (validation) → Dec 16–22 (test) = 1,610 → 1,697 → 1,445 (5161),
1,346 → 1,494 → 1,265 (5059), 1,340 → 1,448 → 1,265 (5259). The test week is 13–15% below the week before.

Same ablation on the **validation** week: in 5161 the weekly lag *helped* (98.49 vs 102.43 without it), so
tuning could not have detected the problem; in 5059 it already *hurt* (96.35 vs 89.99), which tuning on
5161 alone missed.

**Limitations exposed:** (1) a single validation week cannot reveal week-to-week drift; (2) hyperparameters
and features were tuned on one area (5161) and transferred to the others. ARIMA (fixed Fourier shape) and the
LSTM (144-step window) do not use last week's value, which is consistent with them not suffering from this.
