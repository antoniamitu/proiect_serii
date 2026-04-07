# Univariate Time Series Analysis — Food HICP Index (Romania)

**Course:** Time Series Analysis — CSIE, Year III, April 2026
**Topic:** Transmission of the 2021–2023 Energy and Commodity Price Shock into the Romanian Economy

---

## Overview

This repository contains the **univariate analysis** component of the project (requirements 1–6).
The main series is the Romanian Food HICP Index (harmonised consumer price index for food and
non-alcoholic beverages), sourced from IMF STA:CPI, monthly frequency, January 2017 – December 2025.

The analysis covers:
- Deterministic vs. stochastic trend (ADF + KPSS tests)
- Stationarity testing and order of integration
- Exponential smoothing methods (SES, Holt, Holt-Winters)
- SARIMA model identification, estimation, and residual diagnostics
- Point forecasts and 80%/95% confidence intervals
- Out-of-sample forecast comparison (Holt-Winters vs. SARIMA)

---

## Project structure

```
proiect_serii/
│
├── data/
│   ├── raw/
│   │   └── master_dataset.csv        # Input dataset (all series)
│   └── processed/
│       └── food_hicp_series.csv      # Univariate series after preparation
│
├── src/
│   ├── config.py                     # Paths, constants, train/test dates
│   ├── univariate_analysis.py        # Main entry point — runs full pipeline
│   └── utils/
│       ├── load_data.py              # Data loading and preparation
│       ├── descriptive.py            # Descriptive statistics and level/log plots
│       ├── stationarity.py           # ADF and KPSS tests
│       ├── seasonality.py            # STL decomposition, ACF, PACF
│       ├── smoothing_models.py       # SES, Holt, Holt-Winters
│       ├── sarima_tuning.py          # Candidate model grid evaluation
│       ├── sarima_model.py           # Final SARIMA estimation and diagnostics
│       └── evaluation.py             # RMSE, MAE, MAPE metrics
│
├── outputs/
│   ├── figures/                      # All exported plots (PNG, 300 DPI)
│   └── tables/                       # All exported tables (CSV, TXT)
│
├── requirements.txt
└── README.md
```

---

## Data

| Attribute     | Detail                                      |
|---------------|---------------------------------------------|
| Series        | Food HICP Index (food & non-alcoholic bev.) |
| Code          | ROU.HICP.CP01.IX.M                          |
| Source        | IMF STA:CPI                                 |
| Frequency     | Monthly                                     |
| Sample        | January 2017 – December 2025                |
| Observations  | 108                                         |
| Base year     | 2015 = 100                                  |

The master dataset (`master_dataset.csv`) also contains the multivariate series used in the
separate multivariate analysis (requirement 7): `ctot_import`, `pi_sa`, `eur_ron`.

---

## Setup

**Requirements:** Python 3.10+

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## Running the analysis

```bash
python -m src.univariate_analysis
```

All outputs are written to `outputs/figures/` and `outputs/tables/`.
No arguments required — all configuration is in `src/config.py`.

---

## Key results

| Model                      | RMSE  | MAE   | MAPE   |
|----------------------------|-------|-------|--------|
| Holt-Winters               | 8.87  | 7.00  | 4.05%  |
| SARIMA(1,1,1)(0,1,1,12)    | 2.93  | 2.21  | 1.28%  |

Train set: 2017–2023 (84 observations)
Test set: 2024–2025 (24 observations)

The series is integrated of order 1 — I(1). Both ADF and KPSS confirm non-stationarity in
levels and stationarity after first differencing.

---

## Output files

### Figures

| File                        | Content                                      |
|-----------------------------|----------------------------------------------|
| `01_food_hicp_levels.png`   | Series in levels                             |
| `02_food_hicp_log.png`      | Log-transformed series                       |
| `03_stl_decomposition.png`  | STL decomposition (trend, seasonal, residual)|
| `04_acf.png`                | ACF — level series                           |
| `04b_acf_differenced.png`   | ACF — first-differenced series               |
| `05_pacf.png`               | PACF — level series                          |
| `05b_pacf_differenced.png`  | PACF — first-differenced series              |
| `06_hw_forecast.png`        | Holt-Winters forecast vs. test               |
| `07_sarima_forecast.png`    | SARIMA forecast with 80% and 95% CI          |
| `08_residuals_sarima.png`   | SARIMA residuals over time                   |
| `09_qqplot_sarima.png`      | QQ-plot of SARIMA residuals                  |

### Tables

| File                             | Content                                   |
|----------------------------------|-------------------------------------------|
| `descriptive_stats.csv`          | Mean, std, min, max, median, n            |
| `stationarity_results.csv`       | Full ADF and KPSS test results            |
| `stationarity_summary.csv`       | Simplified stationarity conclusions       |
| `smoothing_forecasts.csv`        | SES, Holt, HW forecasts vs. actuals       |
| `smoothing_model_summary.txt`    | Estimated smoothing parameters            |
| `sarima_tuning_results.csv`      | Candidate SARIMA models ranked by RMSE    |
| `sarima_summary.txt`             | Full SARIMAX estimation output            |
| `sarima_forecast.csv`            | SARIMA forecasts with confidence intervals|
| `sarima_residual_diagnostics.csv`| AIC, BIC, Ljung-Box, normality tests      |
| `forecast_accuracy.csv`          | Final RMSE / MAE / MAPE comparison        |