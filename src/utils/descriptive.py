from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    SERIES_COLUMN,
    SERIES_NAME,
    FIGURES_DIR,
    TABLES_DIR,
    FIG_SIZE,
    DPI,
)


def ensure_output_directories() -> None:
    """Create output directories if they do not exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def compute_descriptive_statistics(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for the main univariate series.
    """
    s = series_df[SERIES_COLUMN]

    stats_df = pd.DataFrame(
        {
            "statistic": ["mean", "std", "min", "max", "median", "observations"],
            "value": [
                s.mean(),
                s.std(),
                s.min(),
                s.max(),
                s.median(),
                s.count(),
            ],
        }
    )

    return stats_df


def save_descriptive_statistics(stats_df: pd.DataFrame) -> Path:
    """
    Save descriptive statistics table to outputs/tables.
    """
    ensure_output_directories()
    output_path = TABLES_DIR / "descriptive_stats.csv"
    stats_df.to_csv(output_path, index=False)
    return output_path


def add_log_series(series_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed version of the main series.
    """
    result = series_df.copy()
    result[f"log_{SERIES_COLUMN}"] = np.log(result[SERIES_COLUMN])
    return result


def plot_series_levels(series_df: pd.DataFrame) -> Path:
    """
    Plot the main series in levels.
    """
    ensure_output_directories()

    plt.figure(figsize=FIG_SIZE)
    plt.plot(series_df.index, series_df[SERIES_COLUMN])
    plt.title(f"{SERIES_NAME} in Levels")
    plt.xlabel("Date")
    plt.ylabel(SERIES_NAME)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "01_food_hicp_levels.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path


def plot_series_log(series_df: pd.DataFrame) -> Path:
    """
    Plot the log-transformed main series.
    """
    ensure_output_directories()

    log_column = f"log_{SERIES_COLUMN}"
    if log_column not in series_df.columns:
        raise ValueError(f"Column '{log_column}' not found. Run add_log_series() first.")

    plt.figure(figsize=FIG_SIZE)
    plt.plot(series_df.index, series_df[log_column])
    plt.title(f"log({SERIES_NAME})")
    plt.xlabel("Date")
    plt.ylabel(f"log({SERIES_NAME})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = FIGURES_DIR / "02_food_hicp_log.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()

    return output_path