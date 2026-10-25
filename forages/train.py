"""J2 - Entraînement, suivi MLflow et choix du modèle.

Usage : python -m forages.train
Chaque candidat est un run MLflow (paramètres, métriques de validation croisée et de test, empreinte des
données, commit Git, modèle). Le meilleur candidat en PR-AUC de validation croisée est sauvegardé dans
models/ et enregistré dans le registre MLflow avec l'alias « champion ».
"""

import json
import warnings
import subprocess
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import sklearn
import skops.io
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import forages
from forages import data
from forages.config import (
    CV_FOLDS,
    EXPERIMENT_NAME,
    FEATURES,
    FIGURES_DIR,
    HIST_PATH,
    METADATA_PATH,
    MLFLOW_ARTIFACT_LOCATION,
    MLFLOW_TRACKING_URI,
    MODEL_PATH,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    REPORTS_DIR,
    TARGET_RECALL,
)
from forages.metrics import choose_threshold, compute_metrics
from forages.pipeline import build_pipeline
from forages.reporting import md_table

# Écart de PR-AUC (validation croisée) en dessous duquel deux modèles sont jugés équivalents :
# on garde alors le plus simple, plus facile à expliquer à l'équipe de maintenance.
SIMPLICITY_TOLERANCE = 0.005

warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


def candidates() -> dict:
    """Modèles comparés, du plus simple au plus complexe (l'ordre sert à départager les ex aequo).

    Les arbres n'ont pas besoin de normalisation (numeric="passthrough").
    """
    return {
        "dummy_prior": build_pipeline(DummyClassifier(strategy="prior")),
        "logreg": build_pipeline(LogisticRegression(max_iter=1000)),
        "logreg_balanced_c0.3": build_pipeline(
            LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000)
        ),
        "logreg_splines": build_pipeline(LogisticRegression(C=0.1, max_iter=2000), numeric="splines"),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                min_samples_leaf=10,
                max_features=0.5,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            numeric="passthrough",
        ),
        "hist_gradient_boosting": build_pipeline(
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=8,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=RANDOM_STATE,
            ),
            numeric="passthrough",
        ),
    }


def git_commit() -> str:
    """Commit du code ayant produit le modèle, suffixé de -dirty si des fichiers suivis sont modifiés."""
    cwd = forages.__path__[0]
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=cwd).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no", "--", "forages"],
                                        text=True, cwd=cwd).strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "inconnu"


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_LOCATION)
    mlflow.set_experiment(EXPERIMENT_NAME)


def simple_params(pipe) -> dict:
    params = pipe.named_steps["model"].get_params()
    return {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool)) or v is None}


def evaluate_candidate(name, pipe, X_train, y_train, X_test, y_test, context: dict) -> dict:
    """Validation croisée (sélection + seuil) puis ajustement sur tout le train et mesure sur le test."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(clone(pipe), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = choose_threshold(y_train, oof, TARGET_RECALL)
    cv_metrics = compute_metrics(y_train, oof, threshold)
    folds = [average_precision_score(y_train.iloc[idx], oof[idx]) for _, idx in cv.split(X_train, y_train)]
    cv_metrics["pr_auc_std"] = float(np.std(folds))

    pipe.fit(X_train, y_train)
    test_proba = pipe.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, test_proba, threshold)

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"modele": name, **context})
        mlflow.log_params({"modele": name, "seuil": round(threshold, 4), "rappel_cible": TARGET_RECALL,
                           "cv_folds": CV_FOLDS, "random_state": RANDOM_STATE, **simple_params(pipe)})
        mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_input(mlflow.data.from_pandas(X_train.assign(panne_90j=y_train), source=str(HIST_PATH),
                                                 targets="panne_90j", name="forages_historique_train"),
                         context="training")
        model_info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=infer_signature(X_train.head(), test_proba[:5]),
            input_example=X_train.head(3),
            # Modèle produit par ce script : ses types internes (arbres, etc.) sont de confiance.
            skops_trusted_types=skops.io.get_untrusted_types(data=skops.io.dumps(pipe)),
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
                              f"numpy=={np.__version__}"],
        )
    return {"name": name, "pipe": pipe, "threshold": threshold, "cv": cv_metrics, "test": test_metrics,
            "test_proba": test_proba, "run_id": run.info.run_id, "model_uri": model_info.model_uri}


def select_best(results: list) -> dict:
    """Meilleure PR-AUC de validation croisée ; à moins de SIMPLICITY_TOLERANCE près, le plus simple l'emporte."""
    top = max(r["cv"]["pr_auc"] for r in results)
    return next(r for r in results if r["cv"]["pr_auc"] >= top - SIMPLICITY_TOLERANCE)


def register_champion(best: dict) -> str:
    """Enregistre le modèle retenu dans le registre MLflow et lui donne l'alias « champion »."""
    version = mlflow.register_model(best["model_uri"], REGISTERED_MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, "champion", version.version)
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "seuil", f"{best['threshold']:.4f}")
    client.set_model_version_tag(REGISTERED_MODEL_NAME, version.version, "cv_pr_auc", f"{best['cv']['pr_auc']:.4f}")
    return str(version.version)


def save_figures(results: list, best: dict, X_test, y_test) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in results:
        if r["name"] != "dummy_prior":
            PrecisionRecallDisplay.from_predictions(y_test, r["test_proba"], name=r["name"], ax=ax)
    ax.axhline(y_test.mean(), color="grey", ls="--", label="hasard")
    ax.set_title("Courbes précision-rappel (jeu de test)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "courbes_pr.png", dpi=110)
    plt.close(fig)

    imp = permutation_importance(best["pipe"], X_test, y_test, scoring="average_precision",
                                 n_repeats=10, random_state=RANDOM_STATE)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.barh(np.array(FEATURES)[order], imp.importances_mean[order], xerr=imp.importances_std[order], color="#4C78A8")
    ax.set_title(f"Importance par permutation ({best['name']}, perte de PR-AUC)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "importance_variables.png", dpi=110)
    plt.close(fig)


def write_report(results: list, best: dict, registry_version: str) -> None:
    table = pd.DataFrame(
        [{"modele": r["name"], "cv_pr_auc": r["cv"]["pr_auc"], "cv_pr_auc_std": r["cv"]["pr_auc_std"], "cv_roc_auc": r["cv"]["roc_auc"],
          "seuil": r["threshold"], "test_pr_auc": r["test"]["pr_auc"], "test_roc_auc": r["test"]["roc_auc"],
          "test_rappel": r["test"]["recall"], "test_precision": r["test"]["precision"],
          "test_f2": r["test"]["f2"], "test_brier": r["test"]["brier"], "run_id": r["run_id"]}
         for r in results]
    ).sort_values("cv_pr_auc", ascending=False)
    table.to_csv(REPORTS_DIR / "model_runs.csv", index=False)
    ref = next(r for r in results if r["name"] == "logreg")
    t = best["test"]
    report = f"""# J2 - Choix du modèle

*Généré par `python -m forages.train`. Chaque ligne est un run MLflow (expérience `{EXPERIMENT_NAME}`).*

## Comparaison des candidats

Sélection sur la **PR-AUC en validation croisée {CV_FOLDS} plis** (train uniquement). Le seuil de chaque modèle est
choisi sur ses prédictions hors-pli pour garantir un rappel ≥ {TARGET_RECALL:.0%}, puis appliqué tel quel au test.

{md_table(table.drop(columns="run_id"), index=False)}

## Modèle retenu : `{best['name']}`

- PR-AUC validation croisée : **{best['cv']['pr_auc']:.3f}** (référence logistique : {ref['cv']['pr_auc']:.3f})
- Sur le test (jamais vu pendant la sélection) : PR-AUC {t['pr_auc']:.3f}, ROC-AUC {t['roc_auc']:.3f},
  rappel {t['recall']:.3f}, précision {t['precision']:.3f} au seuil {best['threshold']:.3f}
- Concrètement : sur 100 forages qui vont tomber en panne, environ {t['recall'] * 100:.0f} sont signalés ;
  sur 100 visites déclenchées, environ {t['precision'] * 100:.0f} concernent un forage qui allait réellement tomber en panne.
- Registre MLflow : `{REGISTERED_MODEL_NAME}` version {registry_version}, alias `champion`.

**Argumentaire.**
- Le choix se fait uniquement sur la validation croisée pour ne pas sur-ajuster le jeu de test.
- Règle : meilleure PR-AUC de validation croisée ; si plusieurs modèles sont à moins de {SIMPLICITY_TOLERANCE} les uns
  des autres (bien en dessous de l'écart-type entre plis, ≈ {best['cv']['pr_auc_std']:.3f}), le plus simple l'emporte.
- Les modèles à arbres (forêt aléatoire, gradient boosting) et la logistique à splines n'apportent aucun gain :
  le risque évolue de façon quasi linéaire avec l'âge, l'ancienneté de la maintenance, les pannes passées et la
  fréquentation. La logistique est en outre lisible (un coefficient par variable) et bien calibrée (Brier le plus bas),
  ce qui compte puisque l'API renvoie une probabilité.
- Rééquilibrer les classes (`class_weight="balanced"`) ne change pas le classement (même PR-AUC) mais dégrade la
  calibration : inutile, puisque le seuil est de toute façon ajusté pour le rappel cible.

![Courbes PR](figures/courbes_pr.png)

![Importance](figures/importance_variables.png)
"""
    (REPORTS_DIR / "model_selection.md").write_text(report, encoding="utf-8")


def main() -> None:
    setup_mlflow()
    hist = data.load()
    X_train, X_test, y_train, y_test = data.split(hist)
    context = {"git_commit": git_commit(), "data_sha256": data.file_hash(HIST_PATH)[:16],
               "sklearn_version": sklearn.__version__}

    results = []
    for name, pipe in candidates().items():
        r = evaluate_candidate(name, pipe, X_train, y_train, X_test, y_test, context)
        print(f"{name:<24} cv_pr_auc={r['cv']['pr_auc']:.3f}  test_pr_auc={r['test']['pr_auc']:.3f}  "
              f"test_recall={r['test']['recall']:.3f}  run={r['run_id'][:8]}")
        results.append(r)

    best = select_best(results)
    registry_version = register_champion(best)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipe"], MODEL_PATH)
    trained_at = datetime.now(timezone.utc)
    metadata = {
        "model_name": best["name"],
        "model_version": f"{trained_at:%Y%m%d}-{best['run_id'][:8]}",
        "registry_name": REGISTERED_MODEL_NAME,
        "registry_version": registry_version,
        "mlflow_run_id": best["run_id"],
        "threshold": best["threshold"],
        "target_recall": TARGET_RECALL,
        "features": FEATURES,
        "cv_metrics": best["cv"],
        "test_metrics": best["test"],
        "trained_at": trained_at.isoformat(timespec="seconds"),
        "git_commit": context["git_commit"],
        "data_sha256": context["data_sha256"],
        "sklearn_version": sklearn.__version__,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_figures(results, best, X_test, y_test)
    write_report(results, best, registry_version)
    print(f"\nModèle retenu : {best['name']} -> {MODEL_PATH} (registre v{registry_version}, alias champion)")


if __name__ == "__main__":
    main()
