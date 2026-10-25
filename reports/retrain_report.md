# J5 - Ré-entraînement champion / challenger

*Généré par `python -m forages.retrain`. Ne pas modifier à la main.*

Déclencheur : décision `RE-ENTRAINER` du rapport de dérive ([drift_report.md](drift_report.md)).

- Champion : `20261025-00c9d345` (logreg, seuil 0.276).
- Challenger : même configuration, ré-entraîné sur 3000 forages historiques + 300 forages récents
  (seuil recalculé 0.287) ; registre MLflow version 5, alias `challenger`.
- Arbitre : 300 forages récents, vus par aucun des deux modèles.

| métrique (arbitre) | champion actuel | challenger |
|---|---|---|
| pr_auc | 0.730 | 0.738 |
| roc_auc | 0.789 | 0.792 |
| recall | 0.910 | 0.910 |
| precision | 0.565 | 0.565 |
| f2 | 0.811 | 0.811 |
| brier | 0.187 | 0.185 |
| taux_alerte | 0.720 | 0.720 |

Gain de PR-AUC du challenger : **+0.008** (intervalle de confiance à 95 % par bootstrap apparié, 1000
tirages : [-0.003 ; +0.017] ; le challenger gagne dans 93% des tirages).

**PAS DE PROMOTION** : gain de PR-AUC +0.008 (minimum requis +0.01). Le champion actuel reste en service.

Lecture : la population a changé (plus d'usagers par jour, plus de pannes passées), mais la relation entre ces
variables et le risque de panne est restée la même. Le modèle actuel extrapole correctement : il prédit déjà un risque
plus élevé pour ces forages plus sollicités, et sa calibration reste bonne. Ajouter les données récentes n'apporte
pas un gain suffisant pour justifier un changement de modèle. La dérive doit continuer d'être suivie chaque mois ; c'est la règle de promotion, et non la
seule dérive, qui décide du remplacement du modèle.
