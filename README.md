# Time Series Project - Univariate Analysis

This repository contains the univariate analysis for the **Food HICP Index** series, developed as part of the Time Series project for CSIE, 3rd year.

## Project scope

The analysis covers the univariate requirements:
- deterministic vs stochastic trend
- stationarity testing
- exponential smoothing methods
- SARIMA modeling
- point forecasting and confidence intervals
- forecast comparison

## Data

Input dataset:
- `data/raw/master_dataset.csv`

Main univariate series:
- `food_hicp`

Period:
- January 2017 - December 2025

Frequency:
- Monthly

## Project structure

- `data/` - raw and processed datasets
- `src/` - Python source code
- `outputs/figures/` - exported figures
- `outputs/tables/` - exported tables
- `docs/` - notes and writing support files

## Main scripts

- `src/univariate_analysis.py` - main script
- `src/utils/load_data.py`
- `src/utils/descriptive.py`
- `src/utils/stationarity.py`
- `src/utils/seasonality.py`
- `src/utils/smoothing_models.py`
- `src/utils/sarima_model.py`
- `src/utils/sarima_tuning.py`
- `src/utils/evaluation.py`