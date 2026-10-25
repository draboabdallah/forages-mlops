"""Métriques d'évaluation et choix du seuil de décision.

Métrique principale : PR-AUC (average precision). Le service doit classer les forages par risque
pour planifier un nombre limité de visites, et la classe positive (panne) est minoritaire : la PR-AUC
mesure la qualité de ce classement sur les pannes, là où l'accuracy récompenserait un modèle qui
ne prédit jamais de panne. Le seuil de décision est ensuite fixé pour atteindre un rappel cible
(rater une panne coûte des semaines sans eau ; une visite inutile coûte une journée d'équipe).
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from forages.config import TARGET_RECALL


def choose_threshold(y_true, proba, target_recall: float = TARGET_RECALL) -> float:
    """Plus grand seuil qui garantit un rappel >= target_recall (donc la meilleure précision possible)."""
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    # recall[i] correspond à thresholds[i] ; recall est décroissant quand le seuil augmente.
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return float(thresholds.min())
    return float(thresholds[ok[-1]])


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    y_pred = (proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "taux_alerte": float(y_pred.mean()),
    }
