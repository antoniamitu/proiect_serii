"""
Johansen cointegration test (trace and max eigenvalue).
"""

from __future__ import annotations

import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def johansen_rank(
    endog: np.ndarray,
    det_order: int,
    k_ar_diff: int,
    alpha: float = 0.05,
) -> tuple[int, object]:
    """
    Johansen tests. Returns (cointegration rank r, raw result object).

    Rank from sequential trace test at (1-alpha) critical values (column index 1).
    """
    result = coint_johansen(endog, det_order=det_order, k_ar_diff=k_ar_diff)

    n = endog.shape[1]
    trace = np.asarray(result.lr1)
    t_crit = np.asarray(result.trace_stat_crit_vals)
    meig = np.asarray(result.lr2)
    m_crit = np.asarray(result.max_eig_stat_crit_vals)

    print("\n--- Johansen trace test ---")
    print("H0: rank <= r ; reject if trace statistic > critical value")
    for r in range(n):
        cv95 = t_crit[r, 1]
        print(
            f"  r={r}: trace = {trace[r]:.4f}, crit 95% = {cv95:.4f}, "
            f"reject H0 = {trace[r] > cv95}"
        )

    print("\n--- Johansen max eigenvalue test ---")
    for r in range(n):
        cv95 = m_crit[r, 1]
        print(
            f"  r={r}: max-eig = {meig[r]:.4f}, crit 95% = {cv95:.4f}, "
            f"reject H0 = {meig[r] > cv95}"
        )

    rank = 0
    for r in range(n):
        if trace[r] > t_crit[r, 1]:
            rank = r + 1
        else:
            break
    rank = min(rank, n - 1)
    print(f"\n  Selected cointegration rank (trace, ~{int((1 - alpha) * 100)}%): r = {rank}")
    return rank, result
