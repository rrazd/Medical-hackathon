import pandas as pd
from fastapi.testclient import TestClient

from api.predict import app, initialize_demo


def test_predict_integration():
    df = pd.read_csv("data/reference_cases.csv")
    # initialize demo matcher in-process
    initialize_demo(df, ["age", "erythema"], ["sex", "biologic"])
    client = TestClient(app)
    resp = client.post("/api/predict", json={"age": 42, "erythema": 0.45, "sex": "F"})
    assert resp.status_code == 200
    j = resp.json()
    assert "likelihoods" in j and "matches" in j
    assert isinstance(j["matches"], list)
    # basic structure checks
    for b, info in j["likelihoods"].items():
        assert "p_hat" in info and "ci_lower" in info and "ci_upper" in info
