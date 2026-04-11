from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

from src.config import (
    SERIES_COLUMN,
    SERIES_NAME,
    FIGURES_DIR,
    TABLES_DIR,
    FIG_SIZE,
    DPI,
)


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def run_stl_decomposition(series_df: pd.DataFrame, period: int = 12):
    """
    Run STL decomposition on the level series.
    """
    s = series_df[SERIES_COLUMN].dropna()
    stl = STL(s, period=period, robust=True)
    result = stl.fit()
    return result


def plot_stl_decomposition(series_df: pd.DataFrame, period: int = 12) -> Path:
    """
    Plot STL decomposition of the main series.
    """
    ensure_output_dirs()

    stl_result = run_stl_decomposition(series_df, period=period)
    fig = stl_result.plot()
    fig.set_size_inches(12, 9)
    fig.suptitle(f"STL Decomposition of {SERIES_NAME}", y=1.02)

    output_path = FIGURES_DIR / "03_stl_decomposition.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def plot_acf_series(series_df: pd.DataFrame, lags: int = 36) -> Path:
    """
    Plot ACF for the level series.
    """
    ensure_output_dirs()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_acf(series_df[SERIES_COLUMN].dropna(), lags=lags, ax=ax)
    ax.set_title(f"ACF - {SERIES_NAME}")

    output_path = FIGURES_DIR / "04_acf.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def plot_pacf_series(series_df: pd.DataFrame, lags: int = 36) -> Path:
    """
    Plot PACF for the level series.
    """
    ensure_output_dirs()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_pacf(series_df[SERIES_COLUMN].dropna(), lags=lags, ax=ax, method="ywm")
    ax.set_title(f"PACF - {SERIES_NAME}")

    output_path = FIGURES_DIR / "05_pacf.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def compute_monthly_seasonality_table(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average monthly values to inspect seasonality.
    """
    temp_df = series_df.copy()
    temp_df["month"] = temp_df.index.month

    monthly_avg = (
        temp_df.groupby("month")[SERIES_COLUMN]
        .mean()
        .reset_index()
        .rename(columns={SERIES_COLUMN: "monthly_average"})
    )

    month_labels = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
        9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    monthly_avg["month_name"] = monthly_avg["month"].map(month_labels)

    return monthly_avg


def save_monthly_seasonality_table(monthly_df: pd.DataFrame) -> Path:
    """
    Save monthly average values used to assess seasonality.
    """
    ensure_output_dirs()

    output_path = TABLES_DIR / "monthly_seasonality.csv"
    monthly_df.to_csv(output_path, index=False)
    return output_path


def plot_monthly_seasonality(monthly_df: pd.DataFrame) -> Path:
    """
    Plot average monthly values to make seasonal patterns explicit.
    """
    ensure_output_dirs()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.plot(monthly_df["month_name"], monthly_df["monthly_average"], marker="o")
    ax.set_title(f"Average Monthly Pattern - {SERIES_NAME}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Average index value")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "03b_monthly_seasonality.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def plot_acf_differenced(series_df: pd.DataFrame, lags: int = 36) -> Path:
    """
    Plot ACF for the first-differenced series.
    Used to identify the MA order q for SARIMA.
    """
    ensure_output_dirs()

    diff_col = f"diff_{SERIES_COLUMN}"
    if diff_col not in series_df.columns:
        raise ValueError(f"Column '{diff_col}' not found. Run create_stationarity_variants() first.")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_acf(series_df[diff_col].dropna(), lags=lags, ax=ax)
    ax.set_title(f"ACF - First Difference of {SERIES_NAME}")

    output_path = FIGURES_DIR / "04b_acf_differenced.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def plot_pacf_differenced(series_df: pd.DataFrame, lags: int = 36) -> Path:
    """
    Plot PACF for the first-differenced series.
    Used to identify the AR order p for SARIMA.
    """
    ensure_output_dirs()

    diff_col = f"diff_{SERIES_COLUMN}"
    if diff_col not in series_df.columns:
        raise ValueError(f"Column '{diff_col}' not found. Run create_stationarity_variants() first.")

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    plot_pacf(series_df[diff_col].dropna(), lags=lags, ax=ax, method="ywm")
    ax.set_title(f"PACF - First Difference of {SERIES_NAME}")

    output_path = FIGURES_DIR / "05b_pacf_differenced.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path