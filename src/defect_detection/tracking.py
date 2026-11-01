"""MLflow setup shared by training and evaluation.

View runs with: uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import hashlib
import os
from pathlib import Path

import mlflow

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT = "casting-defects"


def setup(experiment=EXPERIMENT):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment)


def log_split(path=Path("data/splits.csv")):
    """Log which split file a run used, so results are only compared on the same split."""
    digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
    mlflow.log_params({"split_file": str(path), "split_md5": digest})
    mlflow.log_artifact(str(path))
