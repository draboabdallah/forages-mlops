import pytest


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "model_version": "test-0001"}


def test_model_info_expose_la_version(client):
    r = client.get("/v1/model")
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "test-0001"
    assert body["seuil"] == 0.3
    assert body["api_version"]


def test_predict_un_forage(client, forage):
    r = client.post("/v1/predict", json={"forages": [forage]})
    assert r.status_code == 200
    assert r.headers["X-Model-Version"] == "test-0001"
    body = r.json()
    assert body["model_version"] == "test-0001"
    pred = body["predictions"][0]
    assert pred["forage_id"] == "F-0001"
    assert 0 <= pred["probabilite_panne"] <= 1
    assert pred["a_visiter"] == (pred["probabilite_panne"] >= body["seuil"])
    assert pred["niveau_risque"] in {"faible", "moyen", "eleve"}


def test_predict_lot_conserve_l_ordre(client, forage):
    risky = forage | {"forage_id": "risque", "age_ans": 29, "jours_depuis_maintenance": 700, "pannes_12m": 4}
    safe = forage | {"forage_id": "sain", "age_ans": 1, "jours_depuis_maintenance": 10, "pannes_12m": 0,
                     "type_pompe": "solaire", "usagers_jour": 100}
    preds = client.post("/v1/predict", json={"forages": [risky, safe]}).json()["predictions"]
    assert [p["forage_id"] for p in preds] == ["risque", "sain"]
    assert preds[0]["probabilite_panne"] > preds[1]["probabilite_panne"]
    assert preds[0]["niveau_risque"] == "eleve" and preds[1]["niveau_risque"] == "faible"


def test_alias_predict_non_versionne(client, forage):
    r1 = client.post("/predict", json={"forages": [forage]})
    r2 = client.post("/v1/predict", json={"forages": [forage]})
    assert r1.status_code == 200 and r1.json() == r2.json()


@pytest.mark.parametrize(
    "change",
    [
        {"age_ans": -3},
        {"profondeur_m": 0},
        {"debit_m3h": 1000},
        {"pannes_12m": 1.5},
        {"type_pompe": "eolienne"},
        {"region": "Ouest"},
        {"usagers_jour": "beaucoup"},
        {"champ_inconnu": 1},
    ],
)
def test_predict_rejette_les_entrees_invalides(client, forage, change):
    r = client.post("/v1/predict", json={"forages": [forage | change]})
    assert r.status_code == 422


def test_predict_rejette_champ_manquant(client, forage):
    del forage["region"]
    assert client.post("/v1/predict", json={"forages": [forage]}).status_code == 422


def test_predict_rejette_lot_vide(client):
    assert client.post("/v1/predict", json={"forages": []}).status_code == 422


def test_api_degradee_sans_modele(tmp_path, forage):
    import os

    from fastapi.testclient import TestClient

    from forages.api.main import app

    os.environ["MODEL_PATH"] = str(tmp_path / "absent.joblib")
    try:
        with TestClient(app) as c:
            assert c.get("/health").json()["status"] == "degrade"
            assert c.post("/v1/predict", json={"forages": [forage]}).status_code == 503
            assert c.get("/v1/model").status_code == 503
    finally:
        del os.environ["MODEL_PATH"]
