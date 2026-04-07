"""
VAR lag selection, stationary endog construction, VECM FEVD helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def select_var_lag_aic(endog: pd.DataFrame, maxlags: int) -> int:
    """Optimal lag for VAR in stationary coordinates (AIC)."""
    maxlags = max(1, min(maxlags, max(1, len(endog) // (endog.shape[1] + 10))))
    sel = VAR(endog).select_order(maxlags=maxlags)
    p = sel.aic
    if p is None or (isinstance(p, float) and np.isnan(p)):
        p = 1
    else:
        p = int(p)
    p = max(1, p)
    print(f"\n  VAR lag selection (AIC): p = {p}")
    print(f"  (also BIC={sel.bic}, HQIC={sel.hqic}, FPE={sel.fpe})")
    return p


def build_stationary_endog(
    df: pd.DataFrame,
    log_cols: list[str],
    i1_flags: dict[str, bool],
) -> pd.DataFrame:
    """
    VAR on stationary data: Delta log for I(1), level log for I(0).
    Column order follows log_cols.
    """
    series_list = []
    names: list[str] = []
    for c in log_cols:
        if i1_flags.get(c, True):
            nm = f"dlog_{c.replace('log_', '')}"
            series_list.append(df[c].diff())
            names.append(nm)
        else:
            series_list.append(df[c])
            names.append(c)
    out = pd.concat(series_list, axis=1)
    out.columns = names
    return out.dropna()


def fevd_from_vecm(vecm_res, periods: int) -> tuple[np.ndarray, list[str]]:
    """
    FEVD for VECM via orthogonalized MA representation (Cholesky of sigma_u).

    decomp shape: (neqs, periods, neqs).
    """
    orth_irfs = vecm_res.orth_ma_rep(periods)
    orth_slice = orth_irfs[:periods]
    irfs_sq_cum = (orth_slice**2).cumsum(axis=0)

    ma_coefs = vecm_res.ma_rep(periods)
    sigma_u = vecm_res.sigma_u
    k = vecm_res.neqs
    mse = np.zeros((periods, k, k))
    prior = np.zeros((k, k))
    for h in range(periods):
        phi = ma_coefs[h]
        prior = prior + phi @ sigma_u @ phi.T
        mse[h] = prior

    rng = np.arange(k)
    diag_mse = mse[:, rng, rng]

    fevd = np.empty_like(irfs_sq_cum)
    for i in range(periods):
        fevd[i] = (irfs_sq_cum[i].T / diag_mse[i]).T

    decomp = fevd.swapaxes(0, 1)
    names = list(vecm_res.names)
    return decomp, names


def print_fevd_food(
    decomp: np.ndarray,
    names: list[str],
    food_name: str,
    horizons: list[int],
) -> None:
    """Print FEVD shares for the Food equation at selected horizons."""
    if food_name not in names:
        print(f"  Food column {food_name} not in model names {names}")
        return
    i_eq = names.index(food_name)
    print(f"\n  FEVD for equation: {food_name} (rows = shock to variable j)")
    print(f"  Columns: {names}")
    for h in horizons:
        if h > decomp.shape[1]:
            continue
        idx = h - 1
        row = decomp[i_eq, idx, :]
        s = row.sum()
        print(f"\n  Horizon h = {h} (cumulative orthogonalized shares, sum={s:.4f})")
        for j, nm in enumerate(names):
            print(f"    Shock to {nm}: {row[j]:.4f} ({100 * row[j]:.2f}%)")
