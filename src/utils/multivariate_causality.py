"""
Granger causality: bivariate (statsmodels) and VECM Wald tests.
"""

from __future__ import annotations

import contextlib
import io

import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests


def granger_bivariate(
    data: pd.DataFrame,
    caused: str,
    causing: str,
    maxlag: int,
) -> None:
    """
    Columns [caused, causing]. H0: causing does NOT Granger-cause caused.
    """
    pair = data[[caused, causing]].dropna()
    print(f"\n  Pair: does '{causing}' Granger-cause '{caused}'? (H0: no causality)")
    if len(pair) < maxlag + 5:
        print("  [skip] insufficient observations for this lag length.")
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = grangercausalitytests(pair, maxlag=maxlag)
    for lag in range(1, maxlag + 1):
        ftest = res[lag][0]["ssr_ftest"]
        fstat, pval = ftest[0], ftest[1]
        print(
            f"    Lag {lag}: F = {fstat:.4f}, p-value = {pval:.6f}  => "
            f"{'reject H0 (evidence of Granger causality)' if pval < 0.05 else 'fail to reject H0'}"
        )
    best_p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1))
    print(f"  (minimum p-value across lags 1..{maxlag}: {best_p:.6f})")


def granger_vecm_block(vecm_res, caused_names: list[str], causing_names: list[str]) -> None:
    """Multivariate Granger (Wald) consistent with estimated VECM."""
    try:
        caused = caused_names[0] if len(caused_names) == 1 else caused_names
        causing = causing_names[0] if len(causing_names) == 1 else causing_names
        g = vecm_res.test_granger_causality(caused=caused, causing=causing)
        print(f"    VECM Wald Granger test p-value: {g.pvalue:.6f}")
        print(f"    ({g.conclusion})")
    except Exception as exc:  # noqa: BLE001
        print(f"    VECM Granger test failed: {exc}")
