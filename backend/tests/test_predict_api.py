import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


def _valid_png_bytes(size=(640, 512), color=(210, 150, 130)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _intake_form() -> dict:
    return {
        "age": "36",
        "sex": "female",
        "race_ethnicity": "Latina",
        "body_area": "forearms",
        "eczema_duration": "1-3 years",
        "itch_severity": "moderate",
        "atopic_comorbidities": "none",
        "tried_biologics": "no",
        "nonbiologic_treatments": "topical steroids, moisturizer",
        "daily_routine": "desk work and evening runs",
    }


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_predict_returns_real_contract():
    response = client.post(
        "/api/predict",
        data=_intake_form(),
        files={"image": ("baseline.png", _valid_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mock"] is False
    assert body["request_id"].startswith("dm-")
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
    for likelihood in body["likelihoods"]:
        assert 0 <= likelihood["likelihood_pct"] <= 100
        assert likelihood["matched_case_count"] >= 1
    assert body["heatmap"]["overlay_url"] is None
    assert len(body["matched_patients"]) >= 1
    first_match = body["matched_patients"][0]
    assert first_match["before_image_url"].startswith("/api/reference-media/")
    assert first_match["after_image_url"].startswith("/api/reference-media/")


def test_predict_rejects_non_image_upload():
    response = client.post(
        "/api/predict",
        data=_intake_form(),
        files={"image": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400


def test_predict_accepts_small_image_by_upscaling():
    response = client.post(
        "/api/predict",
        data=_intake_form(),
        files={"image": ("tiny.png", _valid_png_bytes(size=(120, 90)), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mock"] is False
    assert [item["biologic"] for item in body["likelihoods"]] == ["Dupixent", "Ebglyss"]


def test_predict_succeeds_without_baseline_severity():
    form = _intake_form()
    form.pop("baseline_severity", None)
    response = client.post(
        "/api/predict",
        data=form,
        files={"image": ("baseline.png", _valid_png_bytes(), "image/png")},
    )

    assert response.status_code == 200
