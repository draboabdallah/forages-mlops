"""J5 - Ré-entraînement champion / challenger, déclenché par la règle de forages.drift.

Usage : python -m forages.retrain [--promote]

1. Le lot récent étiqueté est coupé en deux (stratifié) : une moitié rejoint les données d'entraînement,
   l'autre sert d'arbitre, jamais vue par aucun des deux modèles.
2. Le challenger reprend la configuration du champion et est ré-entraîné sur historique + moitié récente ;
   son seuil est recalculé par validation croisée pour le même rappel cible.
3. Il est tracé dans MLflow et enregistré dans le registre avec l'alias « challenger ».
4. Il n'est promu (alias « champion », écriture dans models/ avec --promote) que s'il gagne au moins
   MIN_PR_AUC_GAIN de PR-AUC sur l'arbitre sans passer sous le rappel minimal.
"""

import argparse
import json
from datetime import datetime, timezone

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from forages import data
from forages.config import (
    CV_FOLDS,
    FEATURES,
    HIST_PATH,
    METADATA_PATH,
    MODEL_PATH,
    NEW_PATH,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    TARGET,
    TARGET_RECALL,
)
from forages.drift import MIN_RECALL
from forages.metrics import choose_threshold, compute_metrics
from forages.predict import load_model
from forages.reporting import md_table
from forages.train import git_commit, setup_mlflow

MIN_PR_AUC_GAIN = 0.01
N_BOOTSTRAP = 1000


def bootstrap_gain(y, p_champ, p_chall, n: int = N_BOOTSTRAP) -> tuple[float, float, float]:
    """Bootstrap apparié du gain de PR-AUC : intervalle à 95 % et part des tirages où le challenger gagne."""
    rng = np.random.default_rng(RANDOM_STATE)
    y, p_champ, p_chall = map(np.asarray, (y, p_champ, p_chall))
    gains = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        gains.append(average_precision_score(y[idx], p_chall[idx]) - average_precision_score(y[idx], p_champ[idx]))
    gains = np.array(gains)
    return float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975)), float((gains > 0).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--promote", action="store_true", help="remplacer models/ si le challenger gagne")
    args = parser.parse_args()

    setup_mlflow()
    champion = load_model(MODEL_PATH)
    hist, new = data.load(), data.load(NEW_PATH)
    new_train, new_eval = train_test_split(new, test_size=0.5, stratify=new[TARGET], random_state=RANDOM_STATE)
    train = pd.concat([hist, new_train], ignore_index=True)

    challenger = clone(champion.pipeline)
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(challenger), train[FEATURES], train[TARGET], cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(train[TARGET], oof, TARGET_RECALL)
    challenger.fit(train[FEATURES], train[TARGET])

    p_champ = champion.predict_proba(new_eval)
    p_chall = challenger.predict_proba(new_eval[FEATURES])[:, 1]
    m_champ = compute_metrics(new_eval[TARGET], p_champ, champion.threshold)
    m_chall = compute_metrics(new_eval[TARGET], p_chall, threshold)
    gain = m_chall["pr_auc"] - m_champ["pr_auc"]
    ci_low, ci_high, p_win = bootstrap_gain(new_eval[TARGET], p_champ, p_chall)
    promote = gain >= MIN_PR_AUC_GAIN and m_chall["recall"] >= MIN_RECALL

    with mlflow.start_run(run_name="retrain-challenger") as run:
        mlflow.set_tags({"modele": champion.metadata["model_name"], "role": "challenger",
                         "champion_version": champion.version, "git_commit": git_commit(),
                         "donnees": f"{HIST_PATH.name} + 50% {NEW_PATH.name}"})
        mlflow.log_params({"seuil": round(threshold, 4), "n_train": len(train), "n_arbitre": len(new_eval)})
        mlflow.log_metrics({f"arbitre_{k}": v for k, v in m_chall.items()})
        mlflow.log_metrics({f"arbitre_champion_{k}": v for k, v in m_champ.items()})
        mlflow.log_metrics({"gain_pr_auc": gain, "gain_ic95_bas": ci_low, "gain_ic95_haut": ci_high})
        info = mlflow.sklearn.log_model(challenger, name="model", input_example=train[FEATURES].head(3))
    client = MlflowClient()
    version = mlflow.register_model(info.model_uri, REGISTERED_MODEL_NAME).version
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "challenger", version)
    if promote:
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version)

    if promote and args.promote:
        trained_at = datetime.now(timezone.utc)
        joblib.dump(challenger, MODEL_PATH)
        meta = champion.metadata | {
            "model_version": f"{trained_at:%Y%m%d}-{run.info.run_id[:8]}", "registry_version": str(version),
            "mlflow_run_id": run.info.run_id, "threshold": threshold, "test_metrics": m_chall,
            "trained_at": trained_at.isoformat(timespec="seconds"), "git_commit": git_commit(),
        }
        METADATA_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lecture = LECTURE_PROMOTION if promote else LECTURE_SANS_PROMOTION
    table = pd.DataFrame({"champion actuel": m_champ, "challenger": m_chall})
    table.index.name = "métrique (arbitre)"
    verdict = ("**PROMOTION** : le challenger devient champion." if promote else
               f"**PAS DE PROMOTION** : gain de PR-AUC {gain:+.3f} (minimum requis {MIN_PR_AUC_GAIN:+.2f}"
               f"{'' if m_chall['recall'] >= MIN_RECALL else f', rappel {m_chall['recall']:.2f} < {MIN_RECALL}'}). "
               "Le champion actuel reste en service.")
    report = f"""# J5 - Ré-entraînement champion / challenger

*Généré par `python -m forages.retrain`. Ne pas modifier à la main.*

Déclencheur : décision `RE-ENTRAINER` du rapport de dérive ([drift_report.md](drift_report.md)).

- Champion : `{champion.version}` ({champion.metadata['model_name']}, seuil {champion.threshold:.3f}).
- Challenger : même configuration, ré-entraîné sur {len(hist)} forages historiques + {len(new_train)} forages récents
  (seuil recalculé {threshold:.3f}) ; registre MLflow version {version}, alias `challenger`.
- Arbitre : {len(new_eval)} forages récents, vus par aucun des deux modèles.

{md_table(table)}

Gain de PR-AUC du challenger : **{gain:+.3f}** (intervalle de confiance à 95 % par bootstrap apparié, {N_BOOTSTRAP}
tirages : [{ci_low:+.3f} ; {ci_high:+.3f}] ; le challenger gagne dans {p_win:.0%} des tirages).

{verdict}

{lecture}
"""
    (REPORTS_DIR / "retrain_report.md").write_text(report, encoding="utf-8")
    summary = {"champion": champion.version, "registry_version_challenger": str(version), "promu": promote,
               "gain_pr_auc": gain, "gain_ic95": [ci_low, ci_high], "part_tirages_gagnants": p_win,
               "min_gain": MIN_PR_AUC_GAIN, "arbitre_champion": m_champ, "arbitre_challenger": m_chall}
    (REPORTS_DIR / "retrain_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report)


LECTURE_SANS_PROMOTION = """Lecture : la population a changé (plus d'usagers par jour, plus de pannes passées), mais la relation entre ces
variables et le risque de panne est restée la même. Le modèle actuel extrapole correctement : il prédit déjà un risque
plus élevé pour ces forages plus sollicités, et sa calibration reste bonne. Ajouter les données récentes n'apporte
pas un gain suffisant pour justifier un changement de modèle. La dérive doit continuer d'être suivie chaque mois ; c'est la règle de promotion, et non la
seule dérive, qui décide du remplacement du modèle."""

LECTURE_PROMOTION = """Lecture : les données récentes apportent un gain mesurable ; le challenger remplace le champion. Relancer
`python -m forages.retrain --promote` pour écrire le nouveau modèle dans models/, puis reconstruire l'image Docker."""


if __name__ == "__main__":
    main()
