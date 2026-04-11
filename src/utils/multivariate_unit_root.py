"""
ADF and KPSS unit-root tests and I(1) classification for multivariate log series.
"""

from __future__ import annotations

import numpy as np
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

    regression='c'  -> level stationarity
    regression='ct' -> trend stationarity
    """
    s = series.dropna()
    stat, p_value, lags, crit = kpss(s, regression=regression, nlags=nlags)
    return {
        "test": f"KPSS_{regression}",
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
    if d["test"].startswith("KPSS"):
        print(f"    KPSS truncation lags: {d.get('lags')}")


def assess_level_evidence(
    adf_level: dict,
    kpss_level_c: dict,
    kpss_level_ct: dict,
    alpha: float = 0.05,
) -> str:
    """
    Summarize evidence about non-stationarity in levels.
    """
    adf_nonstat = adf_level["p_value"] >= alpha
    kpss_c_nonstat = kpss_level_c["p_value"] < alpha
    kpss_ct_nonstat = kpss_level_ct["p_value"] < alpha

    if adf_nonstat and kpss_c_nonstat and kpss_ct_nonstat:
        return "strong evidence of stochastic trend / non-stationary level"
    if adf_nonstat and kpss_c_nonstat and not kpss_ct_nonstat:
        return "possible deterministic trend; level non-stationarity not fully resolved"
    if adf_nonstat or kpss_c_nonstat or kpss_ct_nonstat:
        return "mixed evidence on level non-stationarity"
    return "no strong evidence of level non-stationarity"


def classify_integration(
    adf_level: dict,
    kpss_level_c: dict,
    kpss_level_ct: dict,
    adf_diff: dict,
    kpss_diff: dict,
    alpha: float = 0.05,
) -> tuple[str, bool]:
    """
    Classify whether a series is I(1).

    Preferred pattern:
    - level series: non-stationary
    - first difference: stationary
    """
    adf_nonstat_level = adf_level["p_value"] >= alpha
    kpss_c_nonstat_level = kpss_level_c["p_value"] < alpha
    kpss_ct_nonstat_level = kpss_level_ct["p_value"] < alpha

    strong_level_nonstat = (
        adf_nonstat_level and kpss_c_nonstat_level and kpss_ct_nonstat_level
    )
    mixed_level_nonstat = (
        adf_nonstat_level or kpss_c_nonstat_level or kpss_ct_nonstat_level
    ) and not strong_level_nonstat

    adf_stat_diff = adf_diff["p_value"] < alpha
    kpss_stat_diff = kpss_diff["p_value"] >= alpha
    diff_stationary = adf_stat_diff and kpss_stat_diff

    if strong_level_nonstat and diff_stationary:
        return "I(1)", True

    if mixed_level_nonstat and diff_stationary:
        return "likely I(1) [warning: mixed level-test signals]", True

    return "not I(1) (see level/difference test pattern)", False


def build_integration_table(
    df: pd.DataFrame, var_names: list[str]
) -> tuple[pd.DataFrame, dict[str, bool]]:
    """
    Run ADF and KPSS tests on levels and first differences for each log variable.
    Includes KPSS_c and KPSS_ct on levels for better alignment with the univariate workflow.
    """
    rows = []
    i1_flags: dict[str, bool] = {}

    for v in var_names:
        y = df[v]
        dy = y.diff().dropna()

        adf_l = run_mv_adf(y)
        kpss_l_c = run_mv_kpss(y, regression="c")
        kpss_l_ct = run_mv_kpss(y, regression="ct")
        adf_d = run_mv_adf(dy)
        kpss_d = run_mv_kpss(dy, regression="c")

        print_mv_test_result(v, "levels", adf_l)
        print_mv_test_result(v, "levels", kpss_l_c)
        print_mv_test_result(v, "levels", kpss_l_ct)
        print_mv_test_result(v, "1st difference", adf_d)
        print_mv_test_result(v, "1st difference", kpss_d)

        level_assessment = assess_level_evidence(adf_l, kpss_l_c, kpss_l_ct)
        label, is_i1 = classify_integration(
            adf_l,
            kpss_l_c,
            kpss_l_ct,
            adf_d,
            kpss_d,
        )
        i1_flags[v] = is_i1

        rows.append(
            {
                "variable": v,
                "ADF_level_p": adf_l["p_value"],
                "KPSS_c_level_p": kpss_l_c["p_value"],
                "KPSS_ct_level_p": kpss_l_ct["p_value"],
                "ADF_diff_p": adf_d["p_value"],
                "KPSS_diff_p": kpss_d["p_value"],
                "level_assessment": level_assessment,
                "conclusion": label,
                "i1_flag": is_i1,
            }
        )

        print(f"\n  >>> Level assessment for {v}: {level_assessment}")
        print(f"  >>> Conclusion for {v}: {label}\n")

    return pd.DataFrame(rows), i1_flags