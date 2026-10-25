# Prédiction des pannes de forages à 90 jours

Projet de fin de cours « Apprentissage automatique appliqué à la production ».

**Groupe** : Muhammed Ramadane KOMI · Amir DRABO · Mohamed KABORE. Enseignant : Dr TIEMTORE Souleymane.

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
| Dérive (nouveau mois) | `usagers_jour` +33 % (PSI 0,39) : décision RE-ENTRAINER | [reports/drift_report.md](reports/drift_report.md) |
| Tableau de bord (bonus) | forages classés par risque, filtres, capacité de visites | [reports/dashboard_forages.html](reports/dashboard_forages.html) |
| Ré-entraînement | challenger +0,008 de PR-AUC, sous le seuil de +0,01 : champion conservé | [reports/retrain_report.md](reports/retrain_report.md) |

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
  drift.py                J5 : dérive des données, des prédictions et des performances, règle de décision
  retrain.py              J5 : ré-entraînement champion / challenger
  dashboard.py            bonus : tableau de bord HTML des forages à risque
models/                   modèle retenu (model.joblib) et ses métadonnées (metadata.json)
reports/                  rapports générés et figures
tests/                    tests pytest (données, modèle, API, dérive, garde-fou de performance)
soutenance/               support de soutenance généré par make_slides.py
docs/                     énoncé du projet
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
python -m forages.drift        # J5 : dérive du nouveau mois et décision -> reports/drift_report.md
python -m forages.retrain      # J5 : challenger ré-entraîné, promu seulement s'il gagne -> reports/retrain_report.md
python -m forages.dashboard    # bonus : tableau de bord -> reports/dashboard_forages.html
python soutenance/make_slides.py   # support de soutenance (10 diapositives) -> soutenance/soutenance.pptx
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

L'image a été construite et testée par la CI et en local (conteneur démarré, API interrogée, contrôle de santé `healthy`).

L'image (python:3.12-slim) n'embarque que le code du service, le modèle et les dépendances minimales
(`requirements-api.txt`). Elle tourne sans droits root et déclare un `HEALTHCHECK` sur `/health`.

## Intégration continue

`.github/workflows/ci.yml`, déclenché à chaque push et pull request :

1. tests unitaires et tests de l'API ;
2. ré-entraînement depuis zéro et vérification que les métriques sont identiques à celles du modèle livré ;
3. **garde-fou de performance** (`tests/test_performance.py`) : la CI échoue si PR-AUC < 0,55, ROC-AUC < 0,70,
   rappel < 0,75 ou gain sur le hasard < 0,15 ;
4. rapport de dérive sur le dernier lot, publié avec les autres rapports comme artefact de la CI ;
5. construction de l'image Docker, démarrage du conteneur et appels réels à l'API.

## Surveillance et règle de ré-entraînement

`python -m forages.drift --current <lot.csv>` compare un lot de forages aux données d'entraînement :

- **dérive des variables** : PSI par variable (déciles de la référence), test de Kolmogorov-Smirnov (numériques)
  ou du chi² (catégorielles), part des valeurs hors de la plage vue à l'entraînement ;
- **dérive des prédictions** : PSI des probabilités, part de forages à visiter ;
- **performance**, quand les étiquettes sont connues (90 jours plus tard) : ROC-AUC, rappel, calibration.

| Décision | Condition |
|---|---|
| **RE-ENTRAINER** | PSI d'une variable ≥ 0,25, ou rappel < 0,75, ou ROC-AUC en baisse de plus de 0,05, ou écart de calibration > ±0,05 |
| **SURVEILLER** | PSI d'une variable ou des prédictions entre 0,10 et 0,25, ou plus de 5 % de valeurs hors plage |
| **OK** | sinon |

`--fail-on-retrain` renvoie un code de sortie 1 en cas de RE-ENTRAINER (pour un déclenchement automatique).

Le ré-entraînement (`python -m forages.retrain`) suit un schéma **champion / challenger** : le challenger est appris
sur l'historique et la moitié du lot récent, puis comparé au champion sur l'autre moitié, qu'aucun des deux n'a vue.
Il n'est promu que s'il gagne au moins 0,01 de PR-AUC sans passer sous un rappel de 0,75 (`--promote` écrit alors
le nouveau modèle dans `models/`). Le challenger est enregistré dans le registre MLflow avec l'alias `challenger`.

**Sur le nouveau mois** : la population a changé (+33 % d'usagers par jour, dont 26 % des forages au-delà du
maximum vu à l'entraînement, et plus de pannes passées), ce qui déclenche la règle. Mais le modèle reste performant (ROC-AUC 0,78, rappel 0,89) et bien calibré (écart −0,005) :
il prédit déjà un risque plus élevé pour ces forages plus sollicités. Le challenger ne gagne que +0,008 de PR-AUC
(intervalle de confiance à 95 % : [−0,003 ; +0,017]) et n'est pas promu. Conséquence opérationnelle : la part de
forages signalés passe de 59 % à 72 %, ce qui oblige à prioriser les visites par probabilité décroissante.
Recommandation : comme un quart des forages sont hors de la plage d'entraînement, intégrer les données récentes au
prochain cycle mensuel et relancer la comparaison champion / challenger dès qu'un second mois étiqueté est disponible.

## Tableau de bord des forages à risque (bonus)

`python -m forages.dashboard` génère `reports/dashboard_forages.html`, une page autonome (aucune dépendance
externe) qui s'ouvre dans un navigateur : forages classés par probabilité de panne décroissante, indicateurs clés,
filtres par région, type de pompe et niveau de risque, et un champ « visites possibles » qui montre quelle part
des pannes attendues est couverte par les N premières visites.

## Choix et hypothèses

- **Métrique** : PR-AUC pour le classement des forages (classe positive minoritaire, 36 %), et un seuil qui
  garantit un rappel ≥ 80 %. Rater une panne coûte des semaines sans eau ; une visite inutile coûte une journée.
- **Fuites** : aucune variable ne sépare presque parfaitement les classes ; le prétraitement est dans le pipeline ;
  le test n'intervient pas dans la sélection. Hypothèse à valider avec le métier : `pannes_12m` et
  `jours_depuis_maintenance` sont mesurés à la date de la prédiction.
- **Modèle** : les modèles à arbres et les splines n'apportent rien sur la logistique ; à performance égale, le plus
  simple est retenu (lisible, mieux calibré).
