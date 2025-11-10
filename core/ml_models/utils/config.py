"""
Configuration and global constants for ml_models.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# ML defaults
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

# Supported model types
SUPPORTED_MODELS = [
    "linear_regression",
    "ridge",
    "lasso",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "mlp"
]

# Logging
LOG_LEVEL = "INFO"
