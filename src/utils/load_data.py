from pathlib import Path

import pandas as pd

from src.config import (
    DATA_PATH,
    DATE_COLUMN,
    SERIES_COLUMN,
    FREQUENCY,
    PROCESSED_DATA_DIR,
)


def ensure_directories() -> None:
    """Create processed-data directory if it does not exist."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_master_dataset(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the master dataset from CSV.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)
    return df


def prepare_univariate_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the date and target series columns, convert date to datetime,
    sort values, set date as index, and enforce monthly frequency.
    """
    required_columns = [DATE_COLUMN, SERIES_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    series_df = df[required_columns].copy()
    series_df[DATE_COLUMN] = pd.to_datetime(series_df[DATE_COLUMN], errors="coerce")

    if series_df[DATE_COLUMN].isna().any():
        raise ValueError("Invalid date values found in dataset.")

    series_df = series_df.sort_values(DATE_COLUMN)

    if series_df[SERIES_COLUMN].isna().any():
        raise ValueError(f"Missing values found in target series: {SERIES_COLUMN}")

    series_df = series_df.set_index(DATE_COLUMN)
    series_df = series_df.asfreq(FREQUENCY)

    if series_df[SERIES_COLUMN].isna().any():
        raise ValueError(
            "Missing values appeared after frequency alignment. "
            "Please check whether the dataset is truly monthly and complete."
        )

    return series_df


def save_processed_series(series_df: pd.DataFrame) -> Path:
    """
    Save the prepared univariate series to data/processed.
    """
    ensure_directories()
    output_path = PROCESSED_DATA_DIR / "food_hicp_series.csv"
    series_df.to_csv(output_path)
    return output_path