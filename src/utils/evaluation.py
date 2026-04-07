from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config import TABLES_DIR


def ensure_tables_dir() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compute_forecast_metrics(actual, predicted) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
    }


def compare_models(smoothing_forecast_df: pd.DataFrame, sarima_forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare Holt-Winters and final SARIMA on the test set.
    """
    actual = smoothing_forecast_df["actual"]

    hw_metrics = compute_forecast_metrics(actual, smoothing_forecast_df["hw_forecast"])
    sarima_metrics = compute_forecast_metrics(actual, sarima_forecast_df["sarima_forecast"])

    comparison_df = pd.DataFrame(
        [
            {"model": "Holt-Winters", **hw_metrics},
            {"model": "SARIMA(1,1,1)(0,1,1,12)", **sarima_metrics},
        ]
    )

    return comparison_df


def save_model_comparison(comparison_df: pd.DataFrame) -> Path:
    """
    Save final model comparison table.
    """
    ensure_tables_dir()
    output_path = TABLES_DIR / "forecast_accuracy.csv"
    comparison_df.to_csv(output_path, index=False)
    return output_path