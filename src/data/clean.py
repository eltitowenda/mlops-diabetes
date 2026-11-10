"""Step 3 - Data cleaning.

Applies simple, deterministic cleaning rules to the raw data and writes the
result to the processed data folder, which is what the training step reads.
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ALL_COLUMNS,
    CLEAN_DATA_PATH,
    CLEANING_REPORT_PATH,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    setup_logging,
)

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean ``df``. Returns (clean_dataframe, report)."""
    rows_before = len(df)

    # Keep only the expected columns, in a fixed order
    cleaned = df[ALL_COLUMNS].copy()

    # Ensure numeric types (unparseable values become NaN) and treat +-inf as missing
    for col in cleaned.columns:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)

    # Missing values: drop incomplete rows
    n_missing = int(cleaned.isna().any(axis=1).sum())
    cleaned = cleaned.dropna()

    # Duplicated rows
    n_duplicates = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates()

    # Invalid values: disease progression must be a positive number
    invalid_target = cleaned[TARGET_COLUMN] <= 0
    n_invalid_target = int(invalid_target.sum())
    cleaned = cleaned[~invalid_target]

    cleaned = cleaned.reset_index(drop=True)

    report = {
        "rows_before": int(rows_before),
        "rows_after": int(len(cleaned)),
        "rows_removed_missing": n_missing,
        "rows_removed_duplicates": n_duplicates,
        "rows_removed_invalid_target": n_invalid_target,
    }
    return cleaned, report


def run(
    input_path: Path = RAW_DATA_PATH,
    output_path: Path = CLEAN_DATA_PATH,
    report_path: Path = CLEANING_REPORT_PATH,
) -> Path:
    df = pd.read_csv(input_path)
    cleaned, report = clean_data(df)

    if cleaned.empty:
        raise ValueError("Cleaning removed every row - nothing left to train on.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Cleaning done: %s. Saved to %s", report, output_path)
    return output_path


if __name__ == "__main__":
    setup_logging()
    run()
