"""Model service: loads the champion model from the MLflow Model Registry and
keeps it (and its metadata) in memory."""
import logging
from dataclasses import dataclass, field
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from src.config import (
    CHAMPION_ALIAS,
    FEATURE_COLUMNS,
    REGISTERED_MODEL_NAME,
    RMSE_THRESHOLD,
)
from src.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


@dataclass
class ModelService:
    model: Any = None
    name: str = REGISTERED_MODEL_NAME
    version: str | None = None
    alias: str = CHAMPION_ALIAS
    run_id: str | None = None
    model_type: str | None = None
    metrics: dict = field(default_factory=dict)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        """Load the champion model. Never raises: returns True/False."""
        try:
            setup_mlflow(create_experiment=False)
            client = MlflowClient()
            mv = client.get_model_version_by_alias(self.name, self.alias)
            model = mlflow.pyfunc.load_model(f"models:/{self.name}/{mv.version}")
            run = client.get_run(mv.run_id)

            self.model = model
            self.version = str(mv.version)
            self.run_id = mv.run_id
            self.model_type = run.data.tags.get("model_type")
            self.metrics = {k: float(v) for k, v in run.data.metrics.items()}
            logger.info("Loaded champion model '%s' version %s", self.name, self.version)
            return True
        except Exception:  # model/registry not available -> API answers 503
            logger.exception(
                "Could not load the '%s' model of '%s'. Run the pipeline first "
                "(python -m src.pipeline).", self.alias, self.name,
            )
            self.model = None
            return False

    def info(self) -> dict:
        return {
            "model_name": self.name,
            "model_version": self.version,
            "alias": self.alias,
            "run_id": self.run_id,
            "model_type": self.model_type,
            "metrics": self.metrics,
            "rmse_threshold": RMSE_THRESHOLD,
            "features": FEATURE_COLUMNS,
        }


# Single shared instance used by the API
service = ModelService()
