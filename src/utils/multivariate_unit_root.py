"""
ADF and KPSS unit-root tests and I(1) classification for multivariate log series.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss


def run_mv_adf(series: pd.Series, autolag: str = "AIC") -> dict:
    """
    ADF: H0 = unit root. Reject H0 => evidence of stationarity.
    """
    s = series.dropna()
    res = adfuller(s, autolag=autolag)
    return {
        "test": "ADF",
        "statistic": res[0],
        "p_value": res[1],
        "used_lag": res[2],
        "nobs": res[3],
        "critical_values": res[4],
        "icbest": res[5],
    }


def run_mv_kpss(
    series: pd.Series, regression: str = "c", nlags: str = "auto"
) -> dict:
    """
    KPSS: H0 = stationarity. Reject H0 => non-stationarity.
    regression='c': level stationarity with constant.
    """
    s = series.dropna()
    stat, p_value, lags, crit = kpss(s, regression=regression, nlags=nlags)
    return {
        "test": "KPSS",
        "statistic": stat,
        "p_value": p_value,
        "lags": lags,
        "critical_values": crit,
    }


def print_mv_test_result(name: str, level: str, d: dict) -> None:
    print(f"\n  {name} ({level}) - {d['test']}")
    print(f"    Statistic: {d['statistic']:.6f}")
    print(f"    p-value:   {d['p_value']:.6f}")
    if "critical_values" in d and d["critical_values"] is not None:
        cv = d["critical_values"]
        if isinstance(cv, dict):
            for k, v in sorted(cv.items()):
                print(f"    Critical value {k}: {v:.4f}")
    if d["test"] == "ADF":
        print(f"    ADF used lags: {d.get('used_lag')}, nobs: {d.get('nobs')}")
    if d["test"] == "KPSS":
        print(f"    KPSS truncation lags: {d.get('lags')}")


def classify_integration(
    adf_level: dict,
    kpss_level: dict,
    adf_diff: dict,
    kpss_diff: dict,
    alpha: float = 0.05,
) -> tuple[str, bool]:
    """
    Classify I(1): unit root in levels, stationary in first differences.
    """
    adf_nonstat_level = adf_level["p_value"] >= alpha
    kpss_nonstat_level = kpss_level["p_value"] < alpha
    level_unit_root = adf_nonstat_level and kpss_nonstat_level

    partial = (adf_nonstat_level or kpss_nonstat_level) and not (
        adf_nonstat_level and kpss_nonstat_level
    )

    adf_stat_diff = adf_diff["p_value"] < alpha
    kpss_stat_diff = kpss_diff["p_value"] >= alpha
    diff_stationary = adf_stat_diff and kpss_stat_diff

    is_i1 = bool(level_unit_root and diff_stationary)
    if partial and not level_unit_root:
        is_i1 = diff_stationary and (adf_nonstat_level or kpss_nonstat_level)

    note = "I(1)" if is_i1 else "not I(1) (see test conflict / I(0) / explosive)"
    if partial:
        note += " [warning: mixed level-test signals]"
    return note, is_i1


def build_integration_table(
    df: pd.DataFrame, var_names: list[str]
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """ADF/KPSS on levels and first differences for each log variable."""
    rows = []
    i1_flags: dict[str, bool] = {}
    for v in var_names:
        y = df[v]
        dy = y.diff().dropna()

        adf_l = run_mv_adf(y)
        kpss_l = run_mv_kpss(y)
        adf_d = run_mv_adf(dy)
        kpss_d = run_mv_kpss(dy)

        print_mv_test_result(v, "levels", adf_l)
        print_mv_test_result(v, "levels", kpss_l)
        print_mv_test_result(v, "1st difference", adf_d)
        print_mv_test_result(v, "1st difference", kpss_d)

        label, is_i1 = classify_integration(adf_l, kpss_l, adf_d, kpss_d)
        i1_flags[v] = is_i1
        rows.append(
            {
                "variable": v,
                "ADF_level_p": adf_l["p_value"],
                "KPSS_level_p": kpss_l["p_value"],
                "ADF_diff_p": adf_d["p_value"],
                "KPSS_diff_p": kpss_d["p_value"],
                "conclusion": label,
            }
        )
        print(f"\n  >>> Conclusion for {v}: {label}\n")

    return pd.DataFrame(rows), i1_flags
