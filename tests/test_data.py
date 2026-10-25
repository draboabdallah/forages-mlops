import pytest

from forages import data
from forages.config import FEATURES, NEW_PATH, TARGET, TEST_SIZE


def test_historique_respecte_le_schema(hist):
    assert len(hist) == 3000
    assert list(hist.columns) == FEATURES + [TARGET]


def test_nouveau_mois_respecte_le_schema():
    assert len(data.load(NEW_PATH)) == 600


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda df: df.drop(columns="region"), "Colonnes manquantes"),
        (lambda df: df.assign(age_ans=-1), "hors bornes"),
        (lambda df: df.assign(type_pompe="eolienne"), "Modalités inconnues"),
        (lambda df: df.assign(debit_m3h=None), "Valeurs manquantes"),
        (lambda df: df.assign(panne_90j=2), "binaire"),
    ],
)
def test_validation_rejette_les_donnees_invalides(hist, mutation, message):
    with pytest.raises(ValueError, match=message):
        data.validate(mutation(hist.copy()))


def test_split_stratifie_et_reproductible(hist):
    X_train, X_test, y_train, y_test = data.split(hist)
    assert len(X_test) == int(len(hist) * TEST_SIZE)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert X_train.index.intersection(X_test.index).empty
    X_train2, *_ = data.split(hist)
    assert X_train.index.equals(X_train2.index)
