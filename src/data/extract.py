"""Step 1 - Data extraction.

Loads the Diabetes dataset from scikit-learn and stores it, untouched, in the
raw data folder. In a real project this is where a SQL query, an S3 download
or an API call would live.
"""
import logging
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_diabetes

from src.config import RAW_DATA_PATH, TARGET_COLUMN, setup_logging

logger = logging.getLogger(__name__)


def extract_data() -> pd.DataFrame:
    """Return the diabetes dataset as a DataFrame (features + target)."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df[TARGET_COLUMN] = diabetes.target
    return df


def run(output_path: Path = RAW_DATA_PATH) -> Path:
    df = extract_data()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Raw data saved to %s (shape=%s)", output_path, df.shape)
    return output_path


if __name__ == "__main__":
    setup_logging()
    run()
