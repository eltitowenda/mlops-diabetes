"""Central configuration: paths, constants and thresholds.

This module must stay free of heavy imports (no mlflow, no fastapi) so that
the data scripts and their tests can run with only pandas/scikit-learn.
"""
import logging
import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths (always absolute, so scripts work from any working directory)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes_raw.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "diabetes_clean.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.json"
CLEANING_REPORT_PATH = REPORTS_DIR / "cleaning_report.json"
BEST_RUN_PATH = REPORTS_DIR / "best_run.json"  # hand-off train -> register

# --------------------------------------------------------------------------
# Dataset schema
# --------------------------------------------------------------------------
FEATURE_COLUMNS = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
TARGET_COLUMN = "target"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# A column with more than this fraction of missing/invalid values is a
# *critical* schema problem and stops the pipeline.
MAX_MISSING_RATIO = 0.20

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------
# SQLite backend (needed for the Model Registry) + local artifact folder.
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
)
MLFLOW_EXPERIMENT_NAME = "diabetes-progression"

REGISTERED_MODEL_NAME = "diabetes-progression-regressor"
CHAMPION_ALIAS = "champion"
MODEL_ARTIFACT_PATH = "model"

# --------------------------------------------------------------------------
# Quality gate: a model only becomes champion if its test RMSE is BELOW this.
# (Baseline models reach ~52-55 on this dataset, see README.)
# Can be overridden with the RMSE_THRESHOLD environment variable.
# --------------------------------------------------------------------------
RMSE_THRESHOLD = float(os.getenv("RMSE_THRESHOLD", "60.0"))


def setup_logging() -> None:
    """Configure a consistent log format for every script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
