from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing,
    ExponentialSmoothing,
    Holt,
)

from src.config import (
    SERIES_COLUMN,
    SERIES_NAME,
    TRAIN_END,
    TEST_START,
    FIGURES_DIR,
    TABLES_DIR,
    FIG_SIZE,
    DPI,
)


def ensure_output_directories() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def split_train_test(series_df: pd.DataFrame):
    """
    Split the series into train and test based on config dates.
    """
    train = series_df.loc[:TRAIN_END, SERIES_COLUMN].copy()
    test = series_df.loc[TEST_START:, SERIES_COLUMN].copy()

    if train.empty or test.empty:
        raise ValueError("Train or test split is empty. Check TRAIN_END and TEST_START.")

    return train, test


def fit_ses(train: pd.Series):
    """
    Fit Simple Exponential Smoothing.
    """
    model = SimpleExpSmoothing(train, initialization_method="estimated")
    fitted_model = model.fit(optimized=True)
    return fitted_model


def fit_holt(train: pd.Series):
    """
    Fit Holt's linear trend model.
    """
    model = Holt(train, initialization_method="estimated")
    fitted_model = model.fit(optimized=True)
    return fitted_model


def fit_holt_winters(train: pd.Series, seasonal_periods: int = 12):
    """
    Fit Holt-Winters additive trend + additive seasonality model.
    """
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    fitted_model = model.fit(optimized=True)
    return fitted_model


def generate_forecasts(train: pd.Series, test: pd.Series) -> pd.DataFrame:
    """
    Fit SES, Holt, and Holt-Winters on train and forecast over test horizon.
    """
    ses_model = fit_ses(train)
    holt_model = fit_holt(train)
    hw_model = fit_holt_winters(train)

    horizon = len(test)

    forecast_df = pd.DataFrame(index=test.index)
    forecast_df["actual"] = test
    forecast_df["ses_forecast"] = ses_model.forecast(horizon)
    forecast_df["holt_forecast"] = holt_model.forecast(horizon)
    forecast_df["hw_forecast"] = hw_model.forecast(horizon)

    return forecast_df


def save_smoothing_forecasts(forecast_df: pd.DataFrame) -> Path:
    """
    Save forecast table for smoothing models.
    """
    ensure_output_directories()
    output_path = TABLES_DIR / "smoothing_forecasts.csv"
    forecast_df.to_csv(output_path)
    return output_path


def plot_holt_winters_forecast(train: pd.Series, test: pd.Series, forecast_df: pd.DataFrame) -> Path:
    """
    Plot Holt-Winters forecast against train and test data.
    """
    ensure_output_directories()

    plt.figure(figsize=FIG_SIZE)
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Test")
    plt.plot(forecast_df.index, forecast_df["hw_forecast"], label="Holt-Winters Forecast")
    plt.title(f"{SERIES_NAME} - Holt-Winters Forecast")
    plt.xlabel("Date")
    plt.ylabel(SERIES_NAME)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "06_hw_forecast.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def save_smoothing_model_params(train: pd.Series) -> Path:
    """
    Save a short text summary of smoothing model parameters.
    """
    ensure_output_directories()

    ses_model = fit_ses(train)
    holt_model = fit_holt(train)
    hw_model = fit_holt_winters(train)

    output_path = TABLES_DIR / "smoothing_model_summary.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Simple Exponential Smoothing\n")
        f.write(str(ses_model.params))
        f.write("\n\n")

        f.write("Holt Linear Trend\n")
        f.write(str(holt_model.params))
        f.write("\n\n")

        f.write("Holt-Winters Additive Trend + Additive Seasonality\n")
        f.write(str(hw_model.params))
        f.write("\n")

    return output_path