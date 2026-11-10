"""Step 4 - Model training with MLflow experiment tracking.

Trains several candidate regressors on the cleaned data. Every candidate is an
MLflow run where we log parameters, metrics and artifacts (plus the model
itself). The best run (lowest test RMSE) is written to ``reports/best_run.json``
so the registry step knows which model to register.
"""
import json
import logging

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import (
    BEST_RUN_PATH,
    CLEAN_DATA_PATH,
    FEATURE_COLUMNS,
    MODEL_ARTIFACT_PATH,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    setup_logging,
)
from src.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def get_candidates() -> dict:
    """Candidate models (run name -> estimator)."""
    return {
        "linear_regression": LinearRegression(),
        "ridge_alpha_1": Ridge(alpha=1.0),
        "random_forest_100": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "gradient_boosting_default": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "gradient_boosting_tuned": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
        ),
    }


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run() -> dict:
    df = pd.read_csv(CLEAN_DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train samples: %d | Test samples: %d", len(X_train), len(X_test))

    setup_mlflow()
    best: dict | None = None

    for run_name, model in get_candidates().items():
        with mlflow.start_run(run_name=run_name) as active_run:
            # ---- parameters -------------------------------------------------
            mlflow.set_tags({"model_type": type(model).__name__, "stage": "candidate"})
            params = model.get_params()
            params.update({
                "test_size": TEST_SIZE,
                "split_random_state": RANDOM_STATE,
                "n_train_rows": len(X_train),
                "n_test_rows": len(X_test),
                "features": ",".join(FEATURE_COLUMNS),
            })
            mlflow.log_params(params)

            # ---- training + evaluation -------------------------------------
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            metrics = compute_metrics(y_test, predictions)
            mlflow.log_metrics(metrics)

            # ---- artifacts --------------------------------------------------
            preds_df = pd.DataFrame({"y_true": y_test.values, "y_pred": predictions})
            mlflow.log_text(preds_df.to_csv(index=False), "test_predictions.csv")
            mlflow.log_dict(metrics, "metrics.json")

            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                artifact_path=MODEL_ARTIFACT_PATH,
                signature=signature,
                input_example=X_train.head(3),
            )

            run_id = active_run.info.run_id
            logger.info(
                "%-26s RMSE=%.2f MAE=%.2f R2=%.3f (run %s)",
                run_name, metrics["rmse"], metrics["mae"], metrics["r2"], run_id[:8],
            )

            if best is None or metrics["rmse"] < best["metrics"]["rmse"]:
                best = {
                    "run_id": run_id,
                    "run_name": run_name,
                    "model_type": type(model).__name__,
                    "metrics": metrics,
                }

    BEST_RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    BEST_RUN_PATH.write_text(json.dumps(best, indent=2))
    logger.info("Best run: %s (RMSE=%.2f). Saved to %s",
                best["run_name"], best["metrics"]["rmse"], BEST_RUN_PATH)
    return best


if __name__ == "__main__":
    setup_logging()
    run()
