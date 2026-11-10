"""Step 5 - Model registration, evaluation and champion selection.

1. Reads the best run found by the training step.
2. Registers its model in the MLflow Model Registry (creates a new version).
3. Applies the QUALITY GATE:  test RMSE must be < RMSE_THRESHOLD.
4. Compares against the current champion (if any): the new model only takes
   the ``champion`` alias if it is at least as good (RMSE <=).

If the quality gate fails, the model is still registered (so it is traceable)
but it does NOT get the champion alias and the script exits with an error, so
the CI pipeline turns red.
"""
import json
import logging

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.config import (
    BEST_RUN_PATH,
    CHAMPION_ALIAS,
    MODEL_ARTIFACT_PATH,
    REGISTERED_MODEL_NAME,
    RMSE_THRESHOLD,
    setup_logging,
)
from src.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


class QualityGateError(Exception):
    """Raised when the best model does not satisfy the quality criteria."""


def passes_quality_gate(rmse: float, threshold: float = RMSE_THRESHOLD) -> bool:
    return rmse < threshold


def get_current_champion(client: MlflowClient):
    """Return the model version that holds the champion alias, or None."""
    try:
        return client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None


def run(threshold: float = RMSE_THRESHOLD) -> dict:
    best = json.loads(BEST_RUN_PATH.read_text())
    run_id, rmse = best["run_id"], best["metrics"]["rmse"]

    setup_mlflow(create_experiment=False)
    client = MlflowClient()

    # ---- 1. register the model (always) --------------------------------------
    model_version = mlflow.register_model(f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}", REGISTERED_MODEL_NAME)
    version = model_version.version
    logger.info("Registered '%s' version %s (run %s)", REGISTERED_MODEL_NAME, version, run_id[:8])

    gate_ok = passes_quality_gate(rmse, threshold)
    client.update_model_version(
        REGISTERED_MODEL_NAME,
        version,
        description=f"{best['model_type']} ({best['run_name']}). Test RMSE={rmse:.2f}.",
    )
    for key, value in {
        "rmse": f"{rmse:.4f}",
        "model_type": best["model_type"],
        "rmse_threshold": str(threshold),
        "quality_gate": "passed" if gate_ok else "failed",
    }.items():
        client.set_model_version_tag(REGISTERED_MODEL_NAME, version, key, value)

    # ---- 2. quality gate --------------------------------------------------------
    if not gate_ok:
        raise QualityGateError(
            f"Quality gate FAILED: RMSE {rmse:.2f} is not below threshold {threshold}. "
            f"Version {version} was registered but NOT promoted to '{CHAMPION_ALIAS}'."
        )

    # ---- 3. champion vs. challenger ---------------------------------------------
    current = get_current_champion(client)
    promote = True
    if current is not None and current.version != str(version):
        champion_rmse = client.get_run(current.run_id).data.metrics.get("rmse")
        if champion_rmse is not None and rmse > champion_rmse:
            promote = False
            logger.info(
                "Keeping current champion v%s (RMSE=%.2f): new model is worse (RMSE=%.2f).",
                current.version, champion_rmse, rmse,
            )

    if promote:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, version)
        logger.info("Version %s is now '%s' (RMSE=%.2f < %s).", version, CHAMPION_ALIAS, rmse, threshold)

    return {"version": str(version), "rmse": rmse, "promoted": promote}


if __name__ == "__main__":
    setup_logging()
    run()
