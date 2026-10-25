"""Garde-fou de performance : la CI échoue si le modèle livré dans models/ passe sous les seuils minimaux.

Les seuils sont des engagements métier, pas les scores actuels : ils laissent une marge pour la variance
mais bloquent toute régression nette (mauvais modèle, fuite cassée, prétraitement modifié...).
"""

import pytest
from sklearn.metrics import average_precision_score

from forages import data
from forages.config import MODEL_PATH, NEW_PATH, TARGET_RECALL
from forages.metrics import compute_metrics
from forages.predict import load_model

MIN_PR_AUC = 0.55
MIN_ROC_AUC = 0.70
MIN_RECALL = TARGET_RECALL - 0.05  # tolérance d'échantillonnage sur 600 forages de test
MIN_GAIN_VS_HASARD = 0.15

pytestmark = pytest.mark.performance


@pytest.fixture(scope="module")
def model():
    if not MODEL_PATH.exists():
        pytest.fail(f"Modèle absent ({MODEL_PATH}) : lancer python -m forages.train")
    return load_model(MODEL_PATH)


@pytest.fixture(scope="module")
def test_set(hist):
    _, X_test, _, y_test = data.split(hist)
    return X_test, y_test


def test_metriques_minimales_sur_le_test(model, test_set):
    X_test, y_test = test_set
    m = compute_metrics(y_test, model.predict_proba(X_test), model.threshold)
    assert m["pr_auc"] >= MIN_PR_AUC, m
    assert m["roc_auc"] >= MIN_ROC_AUC, m
    assert m["recall"] >= MIN_RECALL, m


def test_gain_net_sur_le_hasard(model, test_set):
    X_test, y_test = test_set
    assert average_precision_score(y_test, model.predict_proba(X_test)) >= y_test.mean() + MIN_GAIN_VS_HASARD


def test_metadonnees_coherentes_avec_le_modele(model, test_set):
    X_test, y_test = test_set
    recomputed = compute_metrics(y_test, model.predict_proba(X_test), model.threshold)
    assert recomputed["pr_auc"] == pytest.approx(model.metadata["test_metrics"]["pr_auc"], abs=1e-6)
    assert 0 < model.threshold < 1


def test_modele_fonctionne_sur_le_nouveau_mois(model):
    new = data.load(NEW_PATH)
    proba = model.predict_proba(new)
    assert len(proba) == len(new) and ((proba >= 0) & (proba <= 1)).all()
