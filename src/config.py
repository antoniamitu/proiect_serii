from pathlib import Path

# Root project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_PATH = BASE_DIR / "data" / "raw" / "master_dataset.csv"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Output paths
FIGURES_DIR = BASE_DIR / "outputs" / "figures"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

# Main series for univariate analysis
DATE_COLUMN = "date"
SERIES_COLUMN = "food_hicp"
SERIES_NAME = "Food HICP Index"

# Time settings
FREQUENCY = "MS"  # monthly start frequency
TRAIN_END = "2023-12-01"
TEST_START = "2024-01-01"

# Forecast settings
FORECAST_CONFIDENCE_LEVELS = [0.80, 0.95]

# Plot settings
FIG_SIZE = (12, 6)
DPI = 300

# ---------------------------------------------------------------------------
# Multivariate VAR / VECM analysis (canonical CSV column names)
# ---------------------------------------------------------------------------
MV_COL_FOOD = "Food_HICP"
MV_COL_CTOT = "CTOT"
MV_COL_PI = "PI_SA"
MV_COL_EUR = "EUR_RON"

MV_LOG_FOOD = "log_Food_HICP"
MV_LOG_CTOT = "log_CTOT"
MV_LOG_PI = "log_PI_SA"
MV_LOG_EUR = "log_EUR_RON"

# Map lowercase aliases in raw CSV -> canonical names
MV_COLUMN_ALIASES = {
    "food_hicp": MV_COL_FOOD,
    "ctot": MV_COL_CTOT,
    "ctot_import": MV_COL_CTOT,
    "pi_sa": MV_COL_PI,
    "eur_ron": MV_COL_EUR,
}

JOHANSEN_DET_ORDER = 0
VECM_DETERMINISTIC = "ci"

MV_IRF_PERIODS = 12
MV_FEVD_PERIODS = 24
MV_MAX_LAG_ORDER = 12

MULTIVARIATE_FIG_DIR = FIGURES_DIR / "multivariate"