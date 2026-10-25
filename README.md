# Prédiction des pannes de forages à 90 jours

Projet de fin de cours « Apprentissage automatique appliqué à la production ».
Objectif : estimer le risque de panne de chaque forage dans les 90 jours pour planifier les visites préventives.

## Démarrage

```bash
pip install -r requirements.txt
python -m forages.explore      # J1 : exploration, fuites, métrique, modèle de référence -> reports/eda.md
python -m forages.train        # J2 : essais MLflow, choix du modèle -> models/, reports/model_selection.md
mlflow ui --backend-store-uri sqlite:///mlflow.db   # consulter les runs et le registre
```
