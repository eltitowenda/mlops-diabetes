"""Pipeline orchestration: runs the whole MLOps flow in order.

    extract -> validate -> clean -> train -> register/evaluate

Usage (from the repository root):
    python -m src.pipeline
"""
import logging

from src.config import setup_logging
from src.data import clean, extract, validate
from src.registry import register
from src.training import train

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    steps = [
        ("1/5 Data extraction", extract.run),
        ("2/5 Data validation", validate.run),
        ("3/5 Data cleaning", clean.run),
        ("4/5 Model training (MLflow)", train.run),
        ("5/5 Model registration & champion selection", register.run),
    ]
    for name, step in steps:
        logger.info("=== %s ===", name)
        step()
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
