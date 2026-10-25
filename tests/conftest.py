import json
import os

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from forages import data
from forages.api.schemas import EXAMPLE
from forages.pipeline import build_pipeline


@pytest.fixture(scope="session")
def hist():
    return data.load()


@pytest.fixture(scope="session")
def model_path(tmp_path_factory, hist):
    """Modèle léger entraîné pour les tests : l'API est testée indépendamment du modèle livré."""
    X_train, _, y_train, _ = data.split(hist)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    folder = tmp_path_factory.mktemp("model")
    joblib.dump(pipe, folder / "model.joblib")
    metadata = {
        "model_name": "logreg-test",
        "model_version": "test-0001",
        "registry_name": "forages-panne-90j",
        "registry_version": "0",
        "mlflow_run_id": "test",
        "threshold": 0.3,
        "target_recall": 0.8,
        "trained_at": "2026-01-01T00:00:00+00:00",
        "git_commit": "test",
        "test_metrics": {"pr_auc": 0.6},
    }
    (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return folder / "model.joblib"


@pytest.fixture()
def client(model_path):
    os.environ["MODEL_PATH"] = str(model_path)
    from forages.api.main import app

    with TestClient(app) as c:
        yield c
    del os.environ["MODEL_PATH"]


@pytest.fixture()
def forage():
    return dict(EXAMPLE)
