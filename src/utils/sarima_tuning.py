from pathlib import Path

import pandas as pd

from src.config import TABLES_DIR
from src.utils.evaluation import compute_forecast_metrics
from src.utils.sarima_model import fit_sarima, generate_sarima_forecast


def ensure_tables_dir() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def get_candidate_models():
    """
    Small, conservative SARIMA candidate set.
    """
    return [
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 1, 12)},
        {"order": (0, 1, 1), "seasonal_order": (1, 1, 1, 12)},
        {"order": (1, 1, 0), "seasonal_order": (1, 1, 1, 12)},
        {"order": (1, 1, 1), "seasonal_order": (0, 1, 1, 12)},
        {"order": (1, 1, 1), "seasonal_order": (1, 1, 0, 12)},
    ]


def evaluate_sarima_candidates(train: pd.Series, test: pd.Series) -> pd.DataFrame:
    """
    Fit and evaluate a small set of SARIMA candidates.
    """
    results = []

    for candidate in get_candidate_models():
        order = candidate["order"]
        seasonal_order = candidate["seasonal_order"]

        try:
            model = fit_sarima(train, order=order, seasonal_order=seasonal_order)
            forecast_df = generate_sarima_forecast(model, test)

            metrics = compute_forecast_metrics(
                forecast_df["actual"],
                forecast_df["sarima_forecast"],
            )

            results.append(
                {
                    "model": f"SARIMA{order}x{seasonal_order}",
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "AIC": model.aic,
                    "BIC": model.bic,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                }
            )
        except Exception as e:
            results.append(
                {
                    "model": f"SARIMA{order}x{seasonal_order}",
                    "order": str(order),
                    "seasonal_order": str(seasonal_order),
                    "AIC": None,
                    "BIC": None,
                    "RMSE": None,
                    "MAE": None,
                    "MAPE": None,
                    "error": str(e),
                }
            )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["RMSE", "MAE", "MAPE"],
        na_position="last"
    ).reset_index(drop=True)

    return results_df


def save_sarima_tuning_results(results_df: pd.DataFrame) -> Path:
    """
    Save SARIMA tuning results table.
    """
    ensure_tables_dir()
    output_path = TABLES_DIR / "sarima_tuning_results.csv"
    results_df.to_csv(output_path, index=False)
    return output_path