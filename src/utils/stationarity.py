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
    Run stationarity tests on:
    - level series
    - log series
    - first-differenced level series
    - first-differenced log series

    Tests used:
    - ADF for all variants
    - KPSS_c for all variants
    - KPSS_ct only for the level series, to assess trend-stationarity
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

        if variant_name == "level":
            results.append(run_kpss_test(variant_series, variant_name, regression="ct"))

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


def _adf_conclusion(p_value: float, alpha: float) -> str:
    return "stationary" if p_value < alpha else "non-stationary"


def _kpss_conclusion(p_value: float, alpha: float) -> str:
    return "non-stationary" if p_value < alpha else "stationary"


def _combine_adf_kpss(adf_p: float, kpss_p: float, alpha: float) -> str:
    adf_result = _adf_conclusion(adf_p, alpha)
    kpss_result = _kpss_conclusion(kpss_p, alpha)

    if adf_result == "stationary" and kpss_result == "stationary":
        return "likely stationary"
    if adf_result == "non-stationary" and kpss_result == "non-stationary":
        return "likely non-stationary"
    return "mixed evidence"


def _assess_trend_type(
    adf_p: float,
    kpss_c_p: float,
    kpss_ct_p: float | None,
    alpha: float,
) -> str:
    """
    Assess whether the level series is more consistent with:
    - stochastic trend / unit root
    - deterministic trend / trend-stationary behavior
    - mixed evidence

    Interpretation logic:
    - ADF non-rejection + KPSS_c rejection strongly suggests non-stationarity.
    - If KPSS_ct is NOT rejected, this may be compatible with trend-stationarity.
    - If KPSS_ct is also rejected, evidence points more strongly toward a stochastic trend.
    """
    adf_nonstationary = adf_p >= alpha
    kpss_c_nonstationary = kpss_c_p < alpha

    if kpss_ct_p is None:
        if adf_nonstationary and kpss_c_nonstationary:
            return "evidence of non-stationarity; trend form not fully distinguished"
        return "trend form unclear"

    kpss_ct_nonstationary = kpss_ct_p < alpha

    if adf_nonstationary and kpss_c_nonstationary and not kpss_ct_nonstationary:
        return "compatible with deterministic trend (trend-stationary possibility)"
    if adf_nonstationary and kpss_c_nonstationary and kpss_ct_nonstationary:
        return "evidence of stochastic trend (unit root likely)"
    if (not adf_nonstationary) and (not kpss_c_nonstationary):
        return "no strong evidence of trend-driven non-stationarity"
    return "mixed evidence on trend form"


def build_stationarity_summary(results_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Build a simplified interpretation table.

    ADF:
        p < alpha => reject unit root => stationary
    KPSS_c:
        p < alpha => reject level stationarity => non-stationary
    KPSS_ct:
        p < alpha => reject trend stationarity => non-stationary around trend
    """
    summary_rows = []

    for series_name in results_df["series"].unique():
        subset = results_df[results_df["series"] == series_name]

        adf_row = subset[subset["test"] == "ADF"].iloc[0]
        kpss_c_row = subset[subset["test"] == "KPSS_c"].iloc[0]

        kpss_ct_subset = subset[subset["test"] == "KPSS_ct"]
        kpss_ct_row = kpss_ct_subset.iloc[0] if not kpss_ct_subset.empty else None

        adf_p = adf_row["p_value"]
        kpss_c_p = kpss_c_row["p_value"]
        kpss_ct_p = kpss_ct_row["p_value"] if kpss_ct_row is not None else np.nan

        adf_conclusion = _adf_conclusion(adf_p, alpha)
        kpss_c_conclusion = _kpss_conclusion(kpss_c_p, alpha)
        stationarity_conclusion = _combine_adf_kpss(adf_p, kpss_c_p, alpha)

        trend_assessment = ""
        if series_name == "level":
            trend_assessment = _assess_trend_type(
                adf_p=adf_p,
                kpss_c_p=kpss_c_p,
                kpss_ct_p=None if pd.isna(kpss_ct_p) else kpss_ct_p,
                alpha=alpha,
            )

        summary_rows.append(
            {
                "series": series_name,
                "adf_p_value": adf_p,
                "adf_conclusion": adf_conclusion,
                "kpss_c_p_value": kpss_c_p,
                "kpss_c_conclusion": kpss_c_conclusion,
                "kpss_ct_p_value": kpss_ct_p,
                "stationarity_conclusion": stationarity_conclusion,
                "trend_assessment": trend_assessment,
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