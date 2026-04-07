#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multivariate time series analysis: unit roots, Johansen, VAR/VECM, Granger, IRF, FEVD.

Sections (see utils modules):
  1. Imports and setup
  2-3. Load data & logs -> multivariate_load
  4. Plots -> multivariate_plots
  5-6. Unit roots -> multivariate_unit_root
  7. Johansen -> multivariate_cointegration
  8-9. VAR/VECM -> multivariate_models + statsmodels
  10. Granger -> multivariate_causality
  11-12. IRF/FEVD -> multivariate_plots + multivariate_models
  13. Summary (this file)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Allow running as `python src/multivariate_analysis.py` (not only `python -m src.multivariate_analysis`)
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tools.sm_exceptions import InterpolationWarning

from src.config import (
    JOHANSEN_DET_ORDER,
    MV_FEVD_PERIODS,
    MV_IRF_PERIODS,
    MV_LOG_CTOT,
    MV_LOG_EUR,
    MV_LOG_FOOD,
    MV_LOG_PI,
    MV_MAX_LAG_ORDER,
    VECM_DETERMINISTIC,
)
from src.utils.multivariate_causality import granger_bivariate, granger_vecm_block
from src.utils.multivariate_cointegration import johansen_rank
from src.utils.multivariate_load import (
    add_multivariate_log_columns,
    ensure_multivariate_figure_dir,
    find_multivariate_data_path,
    load_multivariate_frame,
    ordered_log_columns,
)
from src.utils.multivariate_models import (
    build_stationary_endog,
    fevd_from_vecm,
    print_fevd_food,
    select_var_lag_aic,
)
from src.utils.multivariate_plots import plot_fevd_stacked, plot_log_series, save_irf_figure
from src.utils.multivariate_unit_root import build_integration_table


def _configure_warnings() -> None:
    warnings.filterwarnings("ignore", category=InterpolationWarning)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="statsmodels.tsa.base.tsa_model",
    )


def main() -> int:
    _configure_warnings()

    print("=" * 72)
    print(" MULTIVARIATE TIME SERIES ANALYSIS - Romania (monthly)")
    print("=" * 72)

    data_path = find_multivariate_data_path()
    print(f"\n[Data file] {data_path}")
    raw, has_eur = load_multivariate_frame(data_path)
    df = add_multivariate_log_columns(raw, has_eur)
    ordered_cols = ordered_log_columns(has_eur)

    for c in ordered_cols:
        if c not in df.columns:
            raise ValueError(f"Missing log column {c} in dataframe.")

    out_dir = ensure_multivariate_figure_dir()

    print("\n" + "-" * 72)
    print(" 4. Plot all series (log levels)")
    print("-" * 72)
    plot_log_series(
        df,
        [c for c in ordered_cols if c in df.columns],
        save_path=out_dir / "log_series.png",
    )

    print("\n" + "-" * 72)
    print(" 5-6. Unit root tests (ADF + KPSS) and integration order")
    print("-" * 72)
    int_table, i1_flags = build_integration_table(df, ordered_cols)
    print("\nIntegration summary table:")
    print(int_table.to_string(index=False))

    all_i1 = all(i1_flags[v] for v in ordered_cols)
    if not all_i1:
        print(
            "\n[Note] Not all series are classified as I(1). Johansen requires I(1) "
            "for standard inference; we skip Johansen and estimate VAR on stationary forms."
        )

    print("\n" + "-" * 72)
    print(" 7. Johansen cointegration test")
    print("-" * 72)
    coint_rank = 0
    if all_i1:
        y = df[ordered_cols].dropna()
        dy = y.diff().dropna()
        max_l = max(2, min(MV_MAX_LAG_ORDER, len(dy) // 5))
        k_ar_diff = select_var_lag_aic(dy, maxlags=max_l)
        k_ar_diff = max(1, k_ar_diff)
        print(f"\n  Johansen / VECM lag order (k_ar_diff): {k_ar_diff}")
        coint_rank, _ = johansen_rank(
            y.values, JOHANSEN_DET_ORDER, k_ar_diff=k_ar_diff
        )
    else:
        k_ar_diff = 2
        print("  Johansen skipped (integration order restriction).")

    print("\n" + "-" * 72)
    print(" 8. Model selection (VAR vs VECM)")
    print("-" * 72)
    if all_i1 and coint_rank >= 1:
        print(f"  Decision: cointegration rank r = {coint_rank} >= 1  =>  estimate VECM.")
        use_vecm = True
    else:
        print("  Decision: r = 0 or not all I(1)  =>  estimate VAR on stationary data")
        print(
            "            (first differences of logs for I(1) series per unit-root tests)."
        )
        use_vecm = False

    print("\n" + "-" * 72)
    print(" 9. Model estimation")
    print("-" * 72)

    vecm_res = None
    var_res = None

    if use_vecm:
        y = df[ordered_cols].dropna()
        model = VECM(
            y,
            k_ar_diff=k_ar_diff,
            coint_rank=coint_rank,
            deterministic=VECM_DETERMINISTIC,
        )
        vecm_res = model.fit()
        print(vecm_res.summary())
    else:
        stat_ordered = build_stationary_endog(df, ordered_cols, i1_flags)
        p_var = select_var_lag_aic(
            stat_ordered,
            maxlags=max(2, min(MV_MAX_LAG_ORDER, len(stat_ordered) // 10)),
        )
        p_var = max(1, p_var)
        var_res = VAR(stat_ordered).fit(p_var)
        print(var_res.summary())

    print("\n" + "-" * 72)
    print("10. Granger causality")
    print("-" * 72)
    dfx = df.copy()
    for c in ordered_cols:
        dfx[f"d_{c}"] = dfx[c].diff()

    dcols = {
        MV_LOG_CTOT: f"d_{MV_LOG_CTOT}",
        MV_LOG_FOOD: f"d_{MV_LOG_FOOD}",
        MV_LOG_PI: f"d_{MV_LOG_PI}",
    }
    if has_eur:
        dcols[MV_LOG_EUR] = f"d_{MV_LOG_EUR}"

    g_lag = k_ar_diff if vecm_res is not None else (var_res.k_ar if var_res else 2)

    print("\n--- Bivariate Granger (first differences of logs; max lag = model lag) ---")
    print("  (a) CTOT -> Food_HICP")
    granger_bivariate(dfx, dcols[MV_LOG_FOOD], dcols[MV_LOG_CTOT], g_lag)
    print("  (b) Food_HICP -> CTOT")
    granger_bivariate(dfx, dcols[MV_LOG_CTOT], dcols[MV_LOG_FOOD], g_lag)
    print("  (c) Food_HICP -> PI_SA")
    granger_bivariate(dfx, dcols[MV_LOG_PI], dcols[MV_LOG_FOOD], g_lag)
    print("  (d) PI_SA -> Food_HICP")
    granger_bivariate(dfx, dcols[MV_LOG_FOOD], dcols[MV_LOG_PI], g_lag)
    if has_eur:
        print("  (e) CTOT -> EUR_RON")
        granger_bivariate(dfx, dcols[MV_LOG_EUR], dcols[MV_LOG_CTOT], g_lag)
        print("  (f) EUR_RON -> Food_HICP")
        granger_bivariate(dfx, dcols[MV_LOG_FOOD], dcols[MV_LOG_EUR], g_lag)

    if vecm_res is not None:
        print("\n--- VECM system Granger (Wald; preferred for cointegrated system) ---")
        print("  CTOT -> Food:")
        granger_vecm_block(vecm_res, [MV_LOG_FOOD], [MV_LOG_CTOT])
        print("  Food -> CTOT:")
        granger_vecm_block(vecm_res, [MV_LOG_CTOT], [MV_LOG_FOOD])
        print("  Food -> PI:")
        granger_vecm_block(vecm_res, [MV_LOG_PI], [MV_LOG_FOOD])
        print("  PI -> Food:")
        granger_vecm_block(vecm_res, [MV_LOG_FOOD], [MV_LOG_PI])
        if has_eur:
            print("  CTOT -> EUR:")
            granger_vecm_block(vecm_res, [MV_LOG_EUR], [MV_LOG_CTOT])
            print("  EUR -> Food:")
            granger_vecm_block(vecm_res, [MV_LOG_FOOD], [MV_LOG_EUR])

    print("\n" + "-" * 72)
    print("11. Impulse response functions (Cholesky; ordering = estimation order)")
    print("-" * 72)
    if has_eur:
        print(
            "  Cholesky order (extended): CTOT -> EUR_RON -> Food_HICP -> PI_SA "
            "(columns in the DataFrame follow this order for estimation)."
        )
    else:
        print("  Cholesky order (baseline): CTOT -> Food_HICP -> PI_SA.")
    print(
        "  Interpretation: a shock to the first variable in the ordering is assumed to "
        "contemporaneously affect all variables below it, but not vice versa; this "
        "reflects treating commodity/import prices as relatively exogenous in the short run."
    )

    if vecm_res is not None:
        irf = vecm_res.irf(MV_IRF_PERIODS)
        save_irf_figure(irf.plot(orth=True, impulse=None, response=None), out_dir, "irf_vecm")
        print(f"  [Saved] IRF plots under {out_dir}")
    else:
        irf = var_res.irf(MV_IRF_PERIODS)
        save_irf_figure(irf.plot(orth=True), out_dir, "irf_var")
        print(f"  [Saved] IRF plots under {out_dir}")

    print("\n" + "-" * 72)
    print("12. Forecast error variance decomposition (focus: Food)")
    print("-" * 72)

    food_eq = MV_LOG_FOOD
    if vecm_res is not None:
        dec, names = fevd_from_vecm(vecm_res, MV_FEVD_PERIODS)
        print_fevd_food(dec, names, food_eq, horizons=[12, 24])
        plot_fevd_stacked(dec, names, food_eq, 24, save_path=out_dir / "fevd_food_vecm.png")
    elif var_res is not None:
        fevd = var_res.fevd(MV_FEVD_PERIODS)
        print(fevd.summary())
        names = list(var_res.names)
        food_candidates = [n for n in names if "Food" in n or "HICP" in n]
        food_col = food_candidates[0] if food_candidates else names[0]
        print_fevd_food(fevd.decomp, names, food_col, horizons=[12, 24])
        fevd.plot()
        plt.savefig(out_dir / "fevd_food_var.png", dpi=150)
        plt.close("all")

    print("\n" + "-" * 72)
    print("13. Final summary (copy-paste friendly)")
    print("-" * 72)
    print(f"  Observations (model sample): {len(df)} months")
    print(f"  Variables (log): {', '.join(ordered_cols)}")
    print(f"  All I(1) [project sense]: {all_i1}")
    print(f"  Cointegration rank r: {coint_rank if all_i1 else 'n/a (Johansen skipped)'}")
    print(f"  Estimated model: {'VECM' if use_vecm else 'VAR in stationary coordinates'}")
    if vecm_res is not None:
        print(f"  VECM k_ar_diff={k_ar_diff}, rank={coint_rank}, det='{VECM_DETERMINISTIC}'")
    if var_res is not None:
        print(f"  VAR lag p={var_res.k_ar} on stationary series.")
    print(f"  Figures folder: {out_dir}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
