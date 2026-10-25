import numpy as np
import pandas as pd
import pytest

from forages.drift import PSI_ALERT, PSI_WARN, decide, psi_categorical, psi_numeric

REF_PERF = {"roc_auc": 0.75, "recall": 0.80}
GOOD_PERF = {"roc_auc": 0.76, "recall": 0.82}


def features(*psis):
    return pd.DataFrame({"psi": psis}, index=[f"v{i}" for i in range(len(psis))])


def test_psi_nul_sur_distributions_identiques():
    x = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert psi_numeric(x, x) == pytest.approx(0, abs=1e-9)


def test_psi_detecte_un_decalage():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 3000))
    assert psi_numeric(ref, pd.Series(rng.normal(0, 1, 3000))) < PSI_WARN
    assert psi_numeric(ref, pd.Series(rng.normal(1, 1, 3000))) > PSI_ALERT


def test_psi_categoriel_et_modalite_nouvelle():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi_categorical(ref, ref) == pytest.approx(0, abs=1e-9)
    assert psi_categorical(ref, pd.Series(["a"] * 50 + ["c"] * 50)) > PSI_ALERT


def test_decision_ok():
    assert decide(features(0.01, 0.02), 0.01, GOOD_PERF, REF_PERF, 0.0)[0] == "OK"


def test_decision_surveiller_sur_derive_moderee():
    assert decide(features(0.15, 0.02), 0.01, GOOD_PERF, REF_PERF, 0.0)[0] == "SURVEILLER"


def test_decision_reentrainer_sur_derive_majeure_sans_etiquettes():
    assert decide(features(0.30), 0.05, None, REF_PERF, None)[0] == "RE-ENTRAINER"


@pytest.mark.parametrize(
    "perf, gap",
    [({"roc_auc": 0.76, "recall": 0.60}, 0.0), ({"roc_auc": 0.65, "recall": 0.80}, 0.0), (GOOD_PERF, 0.12)],
)
def test_decision_reentrainer_sur_baisse_de_performance(perf, gap):
    decision, reasons, _ = decide(features(0.01), 0.01, perf, REF_PERF, gap)
    assert decision == "RE-ENTRAINER" and len(reasons) == 1
