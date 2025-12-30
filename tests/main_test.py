from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to Income Predictor"


def test_post_1():
    body = {
        "age": 40,
        "workclass": "Private",
        "education": "Doctorate",
        "marital_status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 10000,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    r = client.post("/inference", json=body)

    assert r.status_code == 200
    preds = r.json()
    assert float(preds["predictions"][0]) > 0.5


def test_post_2():
    body = {
        "age": 30,
        "workclass": "Private",
        "education": "Doctorate",
        "marital_status": "Married-civ-spouse",
        "occupation": "Tech-support",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 10,
        "hours_per_week": 40,
        "native_country": "United-States",
    }
    r = client.post("/inference", json=body)

    assert r.status_code == 200
    preds = r.json()
    assert float(preds["predictions"][0]) < 0.5
