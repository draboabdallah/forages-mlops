"""Constantes partagées : chemins, schéma des données, paramètres de reproductibilité."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HIST_PATH = DATA_DIR / "forages_historique.csv"
NEW_PATH = DATA_DIR / "forages_nouveau_mois.csv"
MODELS_DIR = ROOT / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))
METADATA_PATH = MODEL_PATH.with_name("metadata.json")
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "panne_90j"
NUM_FEATURES = [
    "age_ans",
    "profondeur_m",
    "debit_m3h",
    "usagers_jour",
    "jours_depuis_maintenance",
    "pannes_12m",
]
CAT_FEATURES = ["type_pompe", "region"]
FEATURES = NUM_FEATURES + CAT_FEATURES
CATEGORIES = {
    "type_pompe": ["manuelle", "solaire", "thermique"],
    "region": ["Centre", "Est", "Nord", "Sud"],
}
# Bornes de plausibilité physique (utilisées pour valider les données et les entrées de l'API).
BOUNDS = {
    "age_ans": (0, 100),
    "profondeur_m": (1, 500),
    "debit_m3h": (0, 100),
    "usagers_jour": (0, 10_000),
    "jours_depuis_maintenance": (0, 3_650),
    "pannes_12m": (0, 50),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
# Objectif métier : ne pas rater plus d'une panne sur cinq.
TARGET_RECALL = 0.80

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
MLFLOW_ARTIFACT_LOCATION = (ROOT / "mlruns").as_uri()
EXPERIMENT_NAME = "forages-panne-90j"
REGISTERED_MODEL_NAME = "forages-panne-90j"
