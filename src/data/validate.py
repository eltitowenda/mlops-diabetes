"""Step 2 - Data validation.

Checks that the raw data respects the expected schema. If any critical problem
is found a ``DataValidationError`` is raised, so the pipeline (and the CI job)
stops *before* training on bad data.
"""
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MAX_MISSING_RATIO,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    VALIDATION_REPORT_PATH,
    setup_logging,
)

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when the dataset has critical schema problems."""


def validate_data(df: pd.DataFrame) -> dict:
    """Validate ``df``. Returns a report dict or raises DataValidationError."""
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Not empty
    if df.empty:
        errors.append("Dataset is empty.")

    # 2. Required columns (features) and target column
    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        errors.append(f"Missing required feature columns: {missing_features}")
    if TARGET_COLUMN not in df.columns:
        errors.append(f"Target column '{TARGET_COLUMN}' is missing.")

    expected = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c in df.columns]

    # 3. Valid data types + 4. critical missing values
    for col in expected:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            coerced = pd.to_numeric(series, errors="coerce")
            invalid_ratio = coerced.isna().mean() if len(series) else 1.0
            if invalid_ratio > MAX_MISSING_RATIO:
                errors.append(
                    f"Column '{col}' is not numeric ({invalid_ratio:.0%} invalid values)."
                )
            else:
                warnings.append(
                    f"Column '{col}' has non-numeric dtype '{series.dtype}'; "
                    "it will be coerced to numeric in the cleaning step."
                )
        else:
            missing_ratio = series.isna().mean() if len(series) else 1.0
            if missing_ratio > MAX_MISSING_RATIO:
                errors.append(
                    f"Column '{col}' has {missing_ratio:.0%} missing values "
                    f"(max allowed {MAX_MISSING_RATIO:.0%})."
                )
            elif missing_ratio > 0:
                warnings.append(
                    f"Column '{col}' has {missing_ratio:.1%} missing values "
                    "(will be handled in cleaning)."
                )

    # Extra, unexpected columns are only a warning
    extra = [c for c in df.columns if c not in FEATURE_COLUMNS + [TARGET_COLUMN]]
    if extra:
        warnings.append(f"Unexpected extra columns will be dropped: {extra}")

    report = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
    }

    if errors:
        raise DataValidationError("; ".join(errors))
    return report


def run(input_path: Path = RAW_DATA_PATH, report_path: Path = VALIDATION_REPORT_PATH) -> dict:
    df = pd.read_csv(input_path)
    try:
        report = validate_data(df)
    except DataValidationError as exc:
        logger.error("Data validation FAILED: %s", exc)
        raise
    for w in report["warnings"]:
        logger.warning(w)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Data validation passed (%d rows). Report: %s", report["rows"], report_path)
    return report


if __name__ == "__main__":
    setup_logging()
    run()
