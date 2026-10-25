"""Contrats d'entrée et de sortie de l'API (validation Pydantic)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forages.config import BOUNDS

EXAMPLE = {
    "forage_id": "F-0001",
    "age_ans": 22,
    "profondeur_m": 96,
    "debit_m3h": 5.4,
    "usagers_jour": 630,
    "jours_depuis_maintenance": 420,
    "pannes_12m": 2,
    "type_pompe": "thermique",
    "region": "Est",
}


def bounded(name: str, description: str):
    low, high = BOUNDS[name]
    return Field(ge=low, le=high, description=f"{description} (entre {low} et {high})")


class Forage(BaseModel):
    """Caractéristiques d'un forage à la date de la prédiction."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    forage_id: str | None = Field(default=None, max_length=64, description="Identifiant libre, renvoyé tel quel")
    age_ans: int = bounded("age_ans", "Âge du forage en années")
    profondeur_m: float = bounded("profondeur_m", "Profondeur en mètres")
    debit_m3h: float = bounded("debit_m3h", "Débit en m³/h")
    usagers_jour: int = bounded("usagers_jour", "Nombre d'usagers par jour")
    jours_depuis_maintenance: int = bounded("jours_depuis_maintenance", "Jours depuis la dernière maintenance")
    pannes_12m: int = bounded("pannes_12m", "Nombre de pannes sur les 12 derniers mois")
    type_pompe: Literal["manuelle", "solaire", "thermique"]
    region: Literal["Centre", "Est", "Nord", "Sud"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forages: list[Forage] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    forage_id: str | None
    probabilite_panne: float = Field(description="Probabilité estimée d'une panne dans les 90 jours")
    a_visiter: bool = Field(description="Vrai si la probabilité dépasse le seuil de décision du modèle")
    niveau_risque: Literal["faible", "moyen", "eleve"]


class PredictionResponse(BaseModel):
    model_version: str
    seuil: float
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    api_version: str
    model_name: str
    model_version: str
    registry_name: str
    registry_version: str
    mlflow_run_id: str
    seuil: float
    rappel_cible: float
    trained_at: str
    git_commit: str
    test_metrics: dict[str, float]


class Health(BaseModel):
    status: Literal["ok", "degrade"]
    model_loaded: bool
    model_version: str | None = None
