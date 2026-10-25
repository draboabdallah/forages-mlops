"""J3 - API de prédiction du risque de panne.

Lancement : uvicorn forages.api.main:app --port 8000   (documentation interactive : /docs)

Versionnement :
- l'API est versionnée dans l'URL (/v1/...) ; /predict reste un alias de /v1/predict ;
- chaque réponse porte la version du modèle (corps et en-tête X-Model-Version), qui renvoie au run MLflow.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Response

import forages
from forages.api.schemas import Health, ModelInfo, Prediction, PredictionRequest, PredictionResponse
from forages.config import MODELS_DIR
from forages.predict import LoadedModel, load_model

logger = logging.getLogger("forages.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", MODELS_DIR / "model.joblib"))
    try:
        app.state.model = load_model(model_path)
        logger.info("Modèle %s chargé depuis %s", app.state.model.version, model_path)
    except FileNotFoundError:
        app.state.model = None
        logger.error("Aucun modèle trouvé à %s : lancer python -m forages.train", model_path)
    yield


app = FastAPI(
    title="API Forages - risque de panne à 90 jours",
    version=forages.__version__,
    description="Estime, pour chaque forage, la probabilité d'une panne dans les 90 jours "
    "afin de planifier les visites préventives.",
    lifespan=lifespan,
)


def get_model() -> LoadedModel:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle indisponible")
    return model


@app.get("/health", response_model=Health, tags=["supervision"])
def health() -> Health:
    model = getattr(app.state, "model", None)
    return Health(status="ok" if model else "degrade", model_loaded=model is not None,
                  model_version=model.version if model else None)


@app.get("/v1/model", response_model=ModelInfo, tags=["modèle"])
def model_info() -> ModelInfo:
    m = get_model().metadata
    return ModelInfo(
        api_version=forages.__version__,
        model_name=m["model_name"],
        model_version=m["model_version"],
        registry_name=m["registry_name"],
        registry_version=m["registry_version"],
        mlflow_run_id=m["mlflow_run_id"],
        seuil=m["threshold"],
        rappel_cible=m["target_recall"],
        trained_at=m["trained_at"],
        git_commit=m["git_commit"],
        test_metrics=m["test_metrics"],
    )


@app.post("/v1/predict", response_model=PredictionResponse, tags=["prédiction"])
def predict(request: PredictionRequest, response: Response) -> PredictionResponse:
    model = get_model()
    df = pd.DataFrame([f.model_dump(exclude={"forage_id"}) for f in request.forages])
    probas = model.predict_proba(df)
    response.headers["X-Model-Version"] = model.version
    return PredictionResponse(
        model_version=model.version,
        seuil=round(model.threshold, 4),
        predictions=[
            Prediction(
                forage_id=f.forage_id,
                probabilite_panne=round(float(p), 4),
                a_visiter=bool(p >= model.threshold),
                niveau_risque=model.risk_level(float(p)),
            )
            for f, p in zip(request.forages, probas)
        ],
    )


# Alias non versionné demandé par le cahier des charges ; pointe toujours vers la version courante.
app.add_api_route("/predict", predict, methods=["POST"], response_model=PredictionResponse, tags=["prédiction"])
