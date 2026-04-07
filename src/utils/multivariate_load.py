"""
Load and prepare the multivariate master dataset: column aliases, dates, logs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BASE_DIR,
    DATA_PATH,
    MV_COL_CTOT,
    MV_COL_EUR,
    MV_COL_FOOD,
    MV_COL_PI,
    MV_COLUMN_ALIASES,
    MV_LOG_CTOT,
    MV_LOG_EUR,
    MV_LOG_FOOD,
    MV_LOG_PI,
    MULTIVARIATE_FIG_DIR,
)


def ensure_multivariate_figure_dir() -> Path:
    """Ensure output directory for multivariate figures exists."""
    MULTIVARIATE_FIG_DIR.mkdir(parents=True, exist_ok=True)
    return MULTIVARIATE_FIG_DIR


def find_multivariate_data_path() -> Path:
    """
    Resolve master_dataset.csv: project data/raw first, then repo root, then CWD.
    """
    candidates = [
        DATA_PATH,
        BASE_DIR / "master_dataset.csv",
        Path.cwd() / "master_dataset.csv",
    ]
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(
        "Could not find master_dataset.csv. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def normalize_multivariate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical project names."""
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for alias, canon in MV_COLUMN_ALIASES.items():
        if alias in lower and lower[alias] != canon:
            rename[lower[alias]] = canon
    if rename:
        df = df.rename(columns=rename)
    return df


def require_multivariate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in (MV_COL_FOOD, MV_COL_CTOT, MV_COL_PI) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Present: {list(df.columns)}"
        )


def load_multivariate_frame(path: Path) -> tuple[pd.DataFrame, bool]:
    """
    Load monthly aligned panel. Returns (dataframe, has_eur).

    Expects Food_HICP, CTOT, PI_SA; EUR_RON is optional.
    """
    df = pd.read_csv(path)
    df = normalize_multivariate_columns(df)
    require_multivariate_columns(df)

    has_eur = MV_COL_EUR in df.columns
    if "date" in df.columns.str.lower():
        date_col = [c for c in df.columns if c.lower() == "date"][0]
        df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
        df = df.sort_values(date_col).set_index(date_col)
    else:
        df.index = pd.RangeIndex(len(df))

    cols = [MV_COL_FOOD, MV_COL_CTOT, MV_COL_PI] + ([MV_COL_EUR] if has_eur else [])
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)

    return df, has_eur


def add_multivariate_log_columns(df: pd.DataFrame, has_eur: bool) -> pd.DataFrame:
    """Natural logs of positive series (indices and optional exchange rate)."""
    out = df.copy()
    for col, lname in [
        (MV_COL_FOOD, MV_LOG_FOOD),
        (MV_COL_CTOT, MV_LOG_CTOT),
        (MV_COL_PI, MV_LOG_PI),
    ]:
        if (out[col] <= 0).any():
            raise ValueError(f"Non-positive values in {col}; cannot take log.")
        out[lname] = np.log(out[col])
    if has_eur:
        if (out[MV_COL_EUR] <= 0).any():
            raise ValueError(f"Non-positive values in {MV_COL_EUR}; cannot take log.")
        out[MV_LOG_EUR] = np.log(out[MV_COL_EUR])
    return out


def ordered_log_columns(has_eur: bool) -> list[str]:
    """Cholesky / estimation order: CTOT, (EUR,) Food, PI."""
    base = [MV_LOG_CTOT, MV_LOG_FOOD, MV_LOG_PI]
    if has_eur:
        return [MV_LOG_CTOT, MV_LOG_EUR, MV_LOG_FOOD, MV_LOG_PI]
    return base
