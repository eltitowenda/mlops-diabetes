"""Small helpers shared by training, registry and API (MLflow setup)."""
import mlflow

from src.config import (
    MLFLOW_ARTIFACTS_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)


def setup_mlflow(create_experiment: bool = True) -> None:
    """Point MLflow to the local SQLite DB and make sure the experiment exists.

    The experiment is created with an *absolute* artifact location so models
    can be loaded from any working directory (scripts, API, CI...).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_registry_uri(MLFLOW_TRACKING_URI)

    if not create_experiment:
        return

    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        MLFLOW_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=MLFLOW_ARTIFACTS_DIR.resolve().as_uri(),
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
