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