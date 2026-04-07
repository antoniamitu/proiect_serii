from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

from src.config import SERIES_COLUMN, TABLES_DIR


def _format_critical_values(critical_values: dict) -> str:
    """
    Convert critical values dictionary to a compact string.
    """
    return " | ".join([f"{key}: {value:.4f}" for key, value in critical_values.items()])


def run_adf_test(series: pd.Series, series_name: str) -> dict:
    """
    Run Augmented Dickey-Fuller test on a series.
    """
    clean_series = series.dropna()

    result = adfuller(clean_series, autolag="AIC")

    return {
        "series": series_name,
        "test": "ADF",
        "test_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3],
        "critical_values": _format_critical_values(result[4]),
    }


def run_kpss_test(series: pd.Series, series_name: str, regression: str = "c") -> dict:
    """
    Run KPSS test on a series.
    regression='c'  -> level stationarity
    regression='ct' -> trend stationarity
    """
    clean_series = series.dropna()

    statistic, p_value, n_lags, critical_values = kpss(
        clean_series,
        regression=regression,
        nlags="auto",
    )

    return {
        "series": series_name,
        "test": f"KPSS_{regression}",
        "test_statistic": statistic,
        "p_value": p_value,
        "lags_used": n_lags,
        "n_obs": len(clean_series),
        "critical_values": _format_critical_values(critical_values),
    }


def create_stationarity_variants(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create level, log, first-difference, and log-difference versions of the series.
    """
    result = series_df.copy()

    log_col = f"log_{SERIES_COLUMN}"
    diff_col = f"diff_{SERIES_COLUMN}"
    log_diff_col = f"diff_log_{SERIES_COLUMN}"

    if log_col not in result.columns:
        result[log_col] = np.log(result[SERIES_COLUMN])

    result[diff_col] = result[SERIES_COLUMN].diff()
    result[log_diff_col] = result[log_col].diff()

    return result


def run_stationarity_suite(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run ADF and KPSS tests on:
    - level series
    - log series
    - first-differenced level series
    - first-differenced log series
    """
    results = []

    variants = {
        "level": series_df[SERIES_COLUMN],
        f"log_{SERIES_COLUMN}": series_df[f"log_{SERIES_COLUMN}"],
        f"diff_{SERIES_COLUMN}": series_df[f"diff_{SERIES_COLUMN}"],
        f"diff_log_{SERIES_COLUMN}": series_df[f"diff_log_{SERIES_COLUMN}"],
    }

    for variant_name, variant_series in variants.items():
        results.append(run_adf_test(variant_series, variant_name))
        results.append(run_kpss_test(variant_series, variant_name, regression="c"))

    results_df = pd.DataFrame(results)
    return results_df


def save_stationarity_results(results_df: pd.DataFrame) -> Path:
    """
    Save full stationarity test results table.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / "stationarity_results.csv"
    results_df.to_csv(output_path, index=False)
    return output_path


def build_stationarity_summary(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Build a simplified interpretation table.

    ADF:
        p < alpha => reject unit root => stationary
    KPSS:
        p < alpha => reject stationarity => non-stationary
    """
    summary_rows = []

    for series_name in results_df["series"].unique():
        subset = results_df[results_df["series"] == series_name]

        adf_row = subset[subset["test"] == "ADF"].iloc[0]
        kpss_row = subset[subset["test"].str.startswith("KPSS")].iloc[0]

        adf_conclusion = "stationary" if adf_row["p_value"] < alpha else "non-stationary"
        kpss_conclusion = "non-stationary" if kpss_row["p_value"] < alpha else "stationary"

        if adf_conclusion == "stationary" and kpss_conclusion == "stationary":
            overall = "likely stationary"
        elif adf_conclusion == "non-stationary" and kpss_conclusion == "non-stationary":
            overall = "likely non-stationary"
        else:
            overall = "mixed evidence"

        summary_rows.append(
            {
                "series": series_name,
                "adf_p_value": adf_row["p_value"],
                "adf_conclusion": adf_conclusion,
                "kpss_p_value": kpss_row["p_value"],
                "kpss_conclusion": kpss_conclusion,
                "overall_conclusion": overall,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def save_stationarity_summary(summary_df: pd.DataFrame) -> Path:
    """
    Save simplified stationarity summary table.
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = TABLES_DIR / "stationarity_summary.csv"
    summary_df.to_csv(output_path, index=False)
    return output_path