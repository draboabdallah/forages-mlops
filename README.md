# Prédiction des pannes de forages à 90 jours

Projet de fin de cours « Apprentissage automatique appliqué à la production ».

Un service de l'hydraulique gère des centaines de forages ; une panne prive un village d'eau pendant des
semaines. Ce dépôt livre un **service de prédiction** qui estime, pour chaque forage, la probabilité d'une panne
dans les 90 jours afin de planifier les visites préventives : modèle reproductible et tracé dans MLflow, exposé par
une API FastAPI, testé, conteneurisé, intégré en continu et surveillé.

## Résultats en bref

| Étape | Résultat | Où le voir |
|---|---|---|
| Métrique | PR-AUC + seuil garantissant un rappel ≥ 80 % | [reports/eda.md](reports/eda.md) |
| Référence | régression logistique : PR-AUC 0,651 en validation croisée (hasard : 0,362) | [reports/eda.md](reports/eda.md) |
| Modèle retenu | régression logistique, PR-AUC test 0,606, rappel 0,81, précision 0,51 | [reports/model_selection.md](reports/model_selection.md) |
| Suivi | 6 runs MLflow, modèle enregistré avec l'alias `champion` | `mlflow ui` |
| API | `/v1/predict`, `/v1/model`, `/health` | http://localhost:8000/docs |

Tous les chiffres de ce dépôt sont produits par les scripts ci-dessous : aucun résultat n'est saisi à la main.

## Organisation du dépôt

```
data/                     données fournies (historique 3 000 forages, nouveau mois 600 forages)
forages/
  config.py               chemins, schéma, graine, rappel cible, MLflow
  data.py                 chargement, validation du schéma, découpage train/test stratifié
  metrics.py              métriques et choix du seuil de décision
  pipeline.py             pipelines scikit-learn (prétraitement + modèle)
  explore.py              J1 : exploration, fuites, modèles de référence
  train.py                J2 : essais MLflow, sélection, registre, sauvegarde du modèle
  predict.py              chargement du modèle et scoring
  api/                    J3 : API FastAPI (schemas.py, main.py)
models/                   modèle retenu (model.joblib) et ses métadonnées (metadata.json)
reports/                  rapports générés et figures
tests/                    tests pytest (données, modèle, API, garde-fou de performance)
Dockerfile                image du service
.github/workflows/ci.yml  intégration continue
```

## Installation

Python 3.12 ou plus récent.

```bash
pip install -r requirements.txt
```

## Reproduire le projet de bout en bout

```bash
python -m forages.explore      # J1 : exploration, fuites, métrique, modèles de référence -> reports/eda.md
python -m forages.train        # J2 : runs MLflow, choix du modèle -> models/, reports/model_selection.md
python -m pytest               # J3/J4 : tests + garde-fou de performance
```

L'entraînement est déterministe (graine fixée, découpage stratifié figé) : relancer `python -m forages.train`
redonne les mêmes métriques, ce que vérifie la CI.

### Suivi MLflow et registre de modèles

Les runs sont stockés dans `mlflow.db` (SQLite) et `mlruns/` (artefacts), régénérés par `python -m forages.train`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Chaque run trace : paramètres du modèle, seuil, métriques de validation croisée (`cv_*`) et de test (`test_*`),
jeu de données d'entraînement, empreinte SHA-256 du fichier, commit Git et modèle sérialisé. Le modèle retenu est
enregistré sous `forages-panne-90j` avec l'alias `champion` (onglet *Models*).

## API

```bash
uvicorn forages.api.main:app --port 8000
```

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/health` | état du service et version du modèle chargé |
| GET | `/v1/model` | version du modèle, run MLflow, seuil, métriques de test |
| POST | `/v1/predict` | prédiction pour un lot de 1 à 1 000 forages |
| POST | `/predict` | alias de `/v1/predict` |

Exemple :

```bash
curl -X POST http://localhost:8000/v1/predict -H "Content-Type: application/json" -d '{"forages":[{"forage_id":"F-0001","age_ans":22,"profondeur_m":96,"debit_m3h":5.4,"usagers_jour":630,"jours_depuis_maintenance":420,"pannes_12m":2,"type_pompe":"thermique","region":"Est"}]}'
```

```json
{"model_version":"20261025-00c9d345","seuil":0.2765,
 "predictions":[{"forage_id":"F-0001","probabilite_panne":0.7455,"a_visiter":true,"niveau_risque":"eleve"}]}
```

- **Validation** : types, bornes physiques (âge 0 à 100 ans, débit 0 à 100 m³/h…), modalités autorisées, champs
  inconnus refusés. Une entrée invalide renvoie une erreur 422 détaillée.
- **Versionnement** : la version de l'API est dans l'URL (`/v1`). La version du modèle (`AAAAMMJJ-<run MLflow>`)
  est renvoyée dans chaque réponse et dans l'en-tête `X-Model-Version`.
- **Décision** : `a_visiter` vaut vrai si la probabilité dépasse le seuil du modèle. `niveau_risque` vaut `eleve`
  au-delà de 0,6, `moyen` entre le seuil et 0,6, `faible` en dessous.
- Sans modèle, l'API démarre en mode dégradé : `/health` le signale et les prédictions renvoient une erreur 503.

## Docker

```bash
docker build -t forages-api .
docker run -p 8000:8000 forages-api
```

L'image (python:3.12-slim) n'embarque que le code du service, le modèle et les dépendances minimales
(`requirements-api.txt`). Elle tourne sans droits root et déclare un `HEALTHCHECK` sur `/health`.

## Intégration continue

`.github/workflows/ci.yml`, déclenché à chaque push et pull request :

1. tests unitaires et tests de l'API ;
2. ré-entraînement depuis zéro et vérification que les métriques sont identiques à celles du modèle livré ;
3. **garde-fou de performance** (`tests/test_performance.py`) : la CI échoue si PR-AUC < 0,55, ROC-AUC < 0,70,
   rappel < 0,75 ou gain sur le hasard < 0,15 ;
4. construction de l'image Docker, démarrage du conteneur et appels réels à l'API.

## Choix et hypothèses

- **Métrique** : PR-AUC pour le classement des forages (classe positive minoritaire, 36 %), et un seuil qui
  garantit un rappel ≥ 80 %. Rater une panne coûte des semaines sans eau ; une visite inutile coûte une journée.
- **Fuites** : aucune variable ne sépare presque parfaitement les classes ; le prétraitement est dans le pipeline ;
  le test n'intervient pas dans la sélection. Hypothèse à valider avec le métier : `pannes_12m` et
  `jours_depuis_maintenance` sont mesurés à la date de la prédiction.
- **Modèle** : les modèles à arbres et les splines n'apportent rien sur la logistique ; à performance égale, le plus
  simple est retenu (lisible, mieux calibré).
