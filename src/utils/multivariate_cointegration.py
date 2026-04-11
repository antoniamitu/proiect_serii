"""
Johansen cointegration test (trace and max eigenvalue).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def _critical_col_from_alpha(alpha: float) -> int:
    """
    Map alpha to the statsmodels Johansen critical-value column:
    col 0 -> 90%
    col 1 -> 95%
    col 2 -> 99%
    """
    if np.isclose(alpha, 0.10):
        return 0
    if np.isclose(alpha, 0.05):
        return 1
    if np.isclose(alpha, 0.01):
        return 2
    raise ValueError("alpha must be one of {0.10, 0.05, 0.01} for Johansen critical values.")


def _sequential_rank(test_stats: np.ndarray, crit_vals: np.ndarray) -> int:
    """
    Sequential Johansen rank selection:
    start from r=0 and stop at the first non-rejection.
    """
    n = len(test_stats)
    rank = 0
    for r in range(n):
        if test_stats[r] > crit_vals[r]:
            rank = r + 1
        else:
            break
    return min(rank, n - 1)


def johansen_rank(
    endog: np.ndarray,
    det_order: int,
    k_ar_diff: int,
    alpha: float = 0.05,
) -> tuple[int, object, pd.DataFrame]:
    """
    Run Johansen tests and return:
    - selected cointegration rank based on the trace test
    - raw statsmodels result object
    - summary table with trace and max-eigenvalue statistics

    The final rank is selected using the sequential trace test at the chosen alpha.
    Max-eigenvalue results are reported as supporting evidence.
    """
    result = coint_johansen(endog, det_order=det_order, k_ar_diff=k_ar_diff)

    crit_col = _critical_col_from_alpha(alpha)
    conf_pct = int((1 - alpha) * 100)

    n = endog.shape[1]
    trace = np.asarray(result.lr1)
    t_crit = np.asarray(result.trace_stat_crit_vals)
    meig = np.asarray(result.lr2)
    m_crit = np.asarray(result.max_eig_stat_crit_vals)

    trace_rank = _sequential_rank(trace, t_crit[:, crit_col])
    maxeig_rank = _sequential_rank(meig, m_crit[:, crit_col])

    rows = []
    for r in range(n):
        rows.append(
            {
                "r": r,
                "trace_stat": trace[r],
                f"trace_crit_{conf_pct}": t_crit[r, crit_col],
                "trace_reject": bool(trace[r] > t_crit[r, crit_col]),
                "maxeig_stat": meig[r],
                f"maxeig_crit_{conf_pct}": m_crit[r, crit_col],
                "maxeig_reject": bool(meig[r] > m_crit[r, crit_col]),
            }
        )

    summary_df = pd.DataFrame(rows)

    print("\n--- Johansen trace test ---")
    print("H0: rank <= r ; reject if trace statistic > critical value")
    for _, row in summary_df.iterrows():
        print(
            f"  r={int(row['r'])}: trace = {row['trace_stat']:.4f}, "
            f"crit {conf_pct}% = {row[f'trace_crit_{conf_pct}']:.4f}, "
            f"reject H0 = {row['trace_reject']}"
        )

    print("\n--- Johansen max eigenvalue test ---")
    for _, row in summary_df.iterrows():
        print(
            f"  r={int(row['r'])}: max-eig = {row['maxeig_stat']:.4f}, "
            f"crit {conf_pct}% = {row[f'maxeig_crit_{conf_pct}']:.4f}, "
            f"reject H0 = {row['maxeig_reject']}"
        )

    print(f"\n  Selected cointegration rank by trace test ({conf_pct}%): r = {trace_rank}")
    print(f"  Selected cointegration rank by max-eigenvalue test ({conf_pct}%): r = {maxeig_rank}")

    if trace_rank == maxeig_rank:
        print("  Trace and max-eigenvalue tests agree on the selected rank.")
    else:
        print("  Warning: trace and max-eigenvalue tests do not fully agree on the selected rank.")

    return trace_rank, result, summary_df