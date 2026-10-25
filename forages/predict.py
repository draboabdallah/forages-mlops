"""Chargement du modèle retenu et scoring d'un lot de forages (utilisé par l'API, la dérive et le tableau de bord)."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from forages.config import FEATURES

# Au-delà de cette probabilité, le forage est prioritaire quel que soit le seuil.
HIGH_RISK_PROBA = 0.6


@dataclass
class LoadedModel:
    pipeline: object
    metadata: dict

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(df[FEATURES])[:, 1]

    def risk_level(self, proba: float) -> str:
        if proba >= max(HIGH_RISK_PROBA, self.threshold):
            return "eleve"
        if proba >= self.threshold:
            return "moyen"
        return "faible"


def load_model(model_path: Path) -> LoadedModel:
    model_path = Path(model_path)
    metadata = json.loads(model_path.with_name("metadata.json").read_text(encoding="utf-8"))
    return LoadedModel(pipeline=joblib.load(model_path), metadata=metadata)
