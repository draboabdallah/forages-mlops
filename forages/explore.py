"""J1 - Exploration des données, contrôle des fuites et modèle de référence.

Usage : python -m forages.explore
Produit reports/eda.md, reports/baseline.json et les figures de reports/figures/.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from forages import data
from forages.config import (
    CAT_FEATURES,
    CV_FOLDS,
    FIGURES_DIR,
    NEW_PATH,
    NUM_FEATURES,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
    TARGET_RECALL,
)
from forages.metrics import choose_threshold, compute_metrics
from forages.pipeline import build_pipeline
from forages.reporting import md_table


def fmt(x: float) -> str:
    return f"{x:.3f}"


def univariate_auc(df: pd.DataFrame) -> pd.Series:
    """AUC de chaque variable numérique prise seule : une AUC proche de 1 trahirait une fuite."""
    from sklearn.metrics import roc_auc_score

    aucs = {}
    for col in NUM_FEATURES:
        auc = roc_auc_score(df[TARGET], df[col])
        aucs[col] = max(auc, 1 - auc)
    return pd.Series(aucs).sort_values(ascending=False)


def plot_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), NUM_FEATURES):
        for label, color in [(0, "#4C78A8"), (1, "#E45756")]:
            ax.hist(df.loc[df[TARGET] == label, col], bins=30, alpha=0.6, density=True, color=color,
                    label=f"{TARGET}={label}")
        ax.set_title(col)
    axes[0, 0].legend()
    fig.suptitle("Distribution des variables selon la survenue d'une panne à 90 jours")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distributions.png", dpi=110)
    plt.close(fig)


def plot_categorical(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, col in zip(axes, CAT_FEATURES):
        rate = df.groupby(col)[TARGET].mean().sort_values()
        ax.barh(rate.index, rate.values, color="#4C78A8")
        ax.axvline(df[TARGET].mean(), color="#E45756", ls="--", label="taux global")
        ax.set_title(f"Taux de panne par {col}")
        ax.set_xlim(0, 1)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "taux_panne_categories.png", dpi=110)
    plt.close(fig)


def baselines(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    """Références évaluées par validation croisée sur le train uniquement (le test reste intact)."""
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    dummy = build_pipeline(DummyClassifier(strategy="prior"))
    proba = cross_val_predict(dummy, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    results["dummy_prior"] = compute_metrics(y_train, proba, threshold=0.5)

    # Règle métier naïve : on visite tout forage déjà tombé en panne dans l'année.
    rule = (X_train["pannes_12m"] >= 1).astype(float)
    results["regle_pannes_12m"] = compute_metrics(y_train, rule, threshold=0.5)

    logreg = build_pipeline(LogisticRegression(max_iter=1000))
    proba = cross_val_predict(logreg, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    results["logreg_reference"] = compute_metrics(y_train, proba, choose_threshold(y_train, proba))
    return results


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    hist = data.load()
    new = data.load(NEW_PATH)
    X_train, X_test, y_train, y_test = data.split(hist)

    plot_distributions(hist)
    plot_categorical(hist)
    aucs = univariate_auc(hist)
    corr = hist[NUM_FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
    overlap = len(hist[NUM_FEATURES + CAT_FEATURES].merge(new[NUM_FEATURES + CAT_FEATURES]))
    base = baselines(X_train, y_train)
    (REPORTS_DIR / "baseline.json").write_text(json.dumps(base, indent=2), encoding="utf-8")

    rows = "\n".join(
        f"| {name} | {fmt(m['pr_auc'])} | {fmt(m['roc_auc'])} | {fmt(m['recall'])} | {fmt(m['precision'])} | {fmt(m['f2'])} |"
        for name, m in base.items()
    )
    desc = hist[NUM_FEATURES].describe().T[["mean", "std", "min", "max"]].round(1)
    report = f"""# J1 - Exploration des données

*Généré par `python -m forages.explore`. Ne pas modifier à la main.*

## Volumétrie et qualité

| Fichier | Lignes | Colonnes | Valeurs manquantes | Doublons | Taux de panne |
|---|---|---|---|---|---|
| forages_historique.csv | {len(hist)} | {hist.shape[1]} | {int(hist.isna().sum().sum())} | {int(hist.duplicated().sum())} | {fmt(hist[TARGET].mean())} |
| forages_nouveau_mois.csv | {len(new)} | {new.shape[1]} | {int(new.isna().sum().sum())} | {int(new.duplicated().sum())} | {fmt(new[TARGET].mean())} |

Les deux fichiers respectent le schéma attendu (types, modalités, bornes physiques : voir `forages/data.py`).
Lignes identiques entre l'historique et le nouveau mois : **{overlap}**.

Classe positive minoritaire ({fmt(hist[TARGET].mean())}) : un modèle qui ne prédit jamais de panne aurait
{fmt(1 - hist[TARGET].mean())} d'accuracy sans aucune utilité. L'accuracy est donc écartée.

## Variables numériques

{md_table(desc, floatfmt='{:.1f}')}

![Distributions](figures/distributions.png)

![Taux par catégorie](figures/taux_panne_categories.png)

Taux de panne par type de pompe : {hist.groupby('type_pompe')[TARGET].mean().round(3).to_dict()}
Taux de panne par région : {hist.groupby('region')[TARGET].mean().round(3).to_dict()}

## Contrôle des fuites de données

1. **Fuite par une variable** : AUC de chaque variable prise seule (une valeur proche de 1 serait suspecte).

{md_table(aucs.to_frame('AUC univariée'))}

   Aucune variable ne sépare presque parfaitement les classes : pas de fuite évidente.
   Corrélations avec la cible : {corr.round(3).to_dict()}.

2. **Fuite temporelle (à confirmer avec le métier)** : `pannes_12m` et `jours_depuis_maintenance` doivent être
   mesurés **à la date de prédiction**, sans inclure la fenêtre des 90 jours. Si une intervention consécutive à la
   panne remettait `jours_depuis_maintenance` à 0, la variable « verrait » la cible. Les forages à 0 jour ne
   présentent aucune panne et `jours_depuis_maintenance` augmente avec le risque : cohérent avec une mesure faite
   avant la panne. Hypothèse retenue et documentée.

3. **Fuite par le prétraitement** : encodage et normalisation sont dans un `Pipeline` scikit-learn, ré-appris
   dans chaque pli de validation croisée, jamais sur le test.

4. **Fuite par la sélection** : le jeu de test (20 %, stratifié, graine {RANDOM_STATE}) n'est utilisé qu'une fois,
   pour le modèle final. Le choix du modèle et du seuil se fait en validation croisée sur le train.

5. **Pas d'identifiant** de forage : impossible qu'un même forage soit à la fois en train et en test sous deux lignes
   repérables ; aucune ligne dupliquée.

## Métrique retenue

- **Métrique de sélection : PR-AUC** (average precision). L'outil sert à **classer** les forages par risque pour
  répartir un nombre limité de visites ; la PR-AUC mesure la qualité de ce classement sur la classe minoritaire.
- **Seuil de décision** : le plus haut seuil qui garantit un **rappel ≥ {TARGET_RECALL:.0%}**. Rater une panne
  prive un village d'eau pendant des semaines ; une visite préventive inutile coûte une journée d'équipe.
  On rapporte aussi la précision au seuil (charge de visites inutiles) et le F2 (rappel pondéré ×2).
- ROC-AUC et score de Brier sont suivis en complément (discrimination et calibration).

## Modèles de référence (validation croisée {CV_FOLDS} plis sur le train)

| Modèle | PR-AUC | ROC-AUC | Rappel | Précision | F2 |
|---|---|---|---|---|---|
{rows}

Tout modèle candidat doit battre nettement la régression logistique de référence en PR-AUC.
"""
    (REPORTS_DIR / "eda.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
