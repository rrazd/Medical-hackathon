from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_predict_returns_mock_final_contract():
    response = client.post(
        "/api/predict",
        data={
            "age": "36",
            "sex": "female",
            "race_ethnicity": "Latina",
            "fitzpatrick_skin_type": "IV",
            "body_area": "forearms",
            "prior_treatments": "topical steroids, moisturizer",
            "baseline_severity": "moderate",
        },
        files={"image": ("baseline.png", b"fake-image-bytes", "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "mock-001"
    assert body["mock"] is True
    assert "not a diagnosis" in body["disclaimer"]
    assert "not stored as an account or EHR record" in body["privacy_notice"]
    assert set(body["patient_features"]) == {
        "erythema_score",
        "lesion_coverage_pct",
        "texture_score",
        "dryness_scaling_score",
        "inflammation_score",
        "affected_body_area_pct",
    }
    assert [item["biologic"] for item in body["likelihoods"]] == ["Dupixent", "Ebglyss"]
    assert body["likelihoods"][0]["likelihood_pct"] == 72
    assert body["likelihoods"][1]["likelihood_pct"] == 64
    assert body["explanation"]["top_contributing_biomarkers"][0]["name"] == "lesion_coverage_pct"
    assert body["heatmap"]["overlay_url"] is None
    assert body["matched_patients"][0]["case_id"] == "MOCK-001"
    assert body["warnings"] == ["Mock response only; no clinical inference has been performed."]


def test_predict_rejects_non_image_upload():
    response = client.post(
        "/api/predict",
        data={
            "age": "36",
            "sex": "female",
            "race_ethnicity": "Latina",
            "fitzpatrick_skin_type": "IV",
            "body_area": "forearms",
            "prior_treatments": "topical steroids",
            "baseline_severity": "moderate",
        },
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
