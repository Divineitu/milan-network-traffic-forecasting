# Forecasting Mobile Network Traffic in Milan

A comparative study of three sequential models for **one-step-ahead (10-minute) forecasting**
of mobile internet traffic, using the Telecom Italia Big Data Challenge dataset for Milan
(1 Nov 2013 – 1 Jan 2014, 10,000 grid squares, 10-minute resolution).

**Research question:** how do different sequential models compare for one-step-ahead mobile
network traffic forecasting, and how does their performance vary across geographical areas
with different traffic characteristics?

**Models compared:** ARIMA with Fourier seasonal terms · LightGBM on lag/calendar features ·
LSTM. Naive persistence and seasonal-naive forecasters are included as reference points.

---

## 1. Requirements

* Python 3.11+ (developed on 3.14), Windows/Linux/macOS
* ~20 GB free disk for the raw data, ~1 GB for processed outputs
* No GPU required; everything runs on a CPU

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

## 2. Getting the data

The dataset is hosted on Harvard Dataverse and is protected by a **guestbook that requires an
email address**, so the download script asks for one. Use your own; it is never stored in the repo.

```bash
python src/data/download.py --email you@example.com          # 62 daily files (19.4 GB) + grid
python src/data/download.py --only telecom --verify          # check every file's MD5
```
The script streams each file in 1 MB blocks, skips files already downloaded (so it can be
re-run after an interruption) and verifies checksums before keeping a file.

## 3. Reproducing the results

Run in this order from the repository root:

| Step | Command | Produces | Approx. time |
|---|---|---|---|
| 1. Memory study | `python src/data/memory_profile.py` | `experiments/results/memory_profile.csv` | 1 min |
| 2. Preprocessing | `python src/data/preprocess.py` | `data/processed/*` | ~8 min |
| 3. Area analysis | `python src/eda/explore_areas.py` | `figures/fig1–fig3`, EDA stats | 1 min |
| 4. Time-series analysis | `python src/eda/temporal_analysis.py` | `figures/fig4–fig5`, ACF/stationarity stats | 3 min |
| 5. Tuning | `python src/experiments/tune.py --model {arima,lgbm,lstm}` | `experiments/results/tuning_*.csv` | 3–20 min each |
| 6. Final evaluation | `python src/experiments/run_final.py` | `figures/fig6–fig7`, metrics, timings, predictions | ~10 min |

Steps 3–6 require step 2. Random seeds are fixed in `src/config.py` (`SEED = 42`).

## 4. Repository layout

```
src/
  config.py                 splits, seeds, feature definitions (single source of truth)
  data/       download.py   Dataverse download with guestbook + MD5 verification
              memory_profile.py  naive vs optimised vs chunked loading measurements
              preprocess.py      19.4 GB of text -> a 341 MB traffic matrix
              loading.py         memory-mapped access to the processed data
  eda/        explore_areas.py   traffic distribution, the five areas, daily/weekly shapes
              temporal_analysis.py  ACF/PACF, MSTL decomposition, ADF/KPSS, predictability
              plotting.py        shared figure style (colourblind-safe palette)
  models/     dataset.py     splits, lag/calendar features, Fourier terms, scaling, windows
              baselines.py   persistence and seasonal-naive reference forecasters
              arima_fourier.py / lgbm.py / lstm.py    the three compared models
  evaluation/ metrics.py     MAE, RMSE, MAPE
  experiments/ tune.py       validation-week hyperparameter experiments
               run_final.py  test-week evaluation, figures, metric and timing tables
experiments/log.md           every experiment with its reasoning
figures/                     all generated figures
```

## 5. Experimental protocol

* **Chronological split:** train 1 Nov – 8 Dec, validation 9–15 Dec, test **16–22 Dec** (1,008 steps).
* **All tuning decisions are made on the validation week.** The test week is used once, at the end.
* **One-step-ahead from the true history:** predicting x(t+1) uses observations up to t, never
  the model's own earlier predictions.
* **Scaling is fitted on training data only** (the LSTM's min–max scaler), to prevent leakage.
* Models are refitted on train + validation before the test week.
* Timings are measured with `time.perf_counter`; inference is averaged over 5 repetitions and
  reported as the mean over the three areas.

**Hardware used for the reported timings:** Intel Core i7-8665U (4 cores, 1.9 GHz), 16 GB RAM,
Windows 11, CPU only, Python 3.14, PyTorch 2.13 (CPU build).

## 6. Data licence and attribution

The dataset is released under the **ODbL 1.0** licence. Any use must be accompanied by a link
reading "from BigDataChallenge contest"
(http://www.telecomitalia.com/tit/en/bigdatachallenge.html).

Dataset: G. Barlacchi *et al.*, "A multi-source dataset of urban life in the city of Milan and
the Province of Trentino," *Sci. Data*, vol. 2, 150055, 2015, doi: 10.1038/sdata.2015.55.
Files: doi:10.7910/DVN/EGZHFV (telecom activity) and doi:10.7910/DVN/QJWLFU (Milan grid).

The raw and processed data are **not** committed to this repository; re-create them with steps 2–3.

## 7. Report and video

* Report: `report/` (PDF)
* Video: *link to be added*
