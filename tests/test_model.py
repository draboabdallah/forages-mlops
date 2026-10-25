import numpy as np
from sklearn.linear_model import LogisticRegression

from forages import data
from forages.metrics import choose_threshold, compute_metrics
from forages.pipeline import build_pipeline


def test_choose_threshold_atteint_le_rappel_cible():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    proba = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    threshold = choose_threshold(y, proba, target_recall=0.8)
    assert compute_metrics(y, proba, threshold)["recall"] >= 0.8


def test_metriques_cas_parfait():
    y = np.array([0, 0, 1, 1])
    m = compute_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]), threshold=0.5)
    assert m["pr_auc"] == 1.0 and m["recall"] == 1.0 and m["precision"] == 1.0


def test_pipeline_probabilites_valides_et_categorie_inconnue(hist):
    X_train, X_test, y_train, _ = data.split(hist)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    # Une modalité jamais vue ne doit pas faire planter le pipeline (handle_unknown="ignore").
    odd = X_test.head(1).assign(region="Ouest")
    assert pipe.predict_proba(odd).shape == (1, 2)


def test_entrainement_reproductible(hist):
    X_train, X_test, y_train, _ = data.split(hist)
    p1 = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train).predict_proba(X_test)
    p2 = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train).predict_proba(X_test)
    np.testing.assert_allclose(p1, p2)


def test_modele_bat_la_reference_naive(hist):
    X_train, X_test, y_train, y_test = data.split(hist)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    assert compute_metrics(y_test, proba, 0.5)["pr_auc"] > y_test.mean() + 0.15
