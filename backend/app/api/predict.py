from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.predict import PredictResponse
from app.services.mock_predict import build_mock_predict_response

router = APIRouter(prefix="/api", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    race_ethnicity: str = Form(...),
    fitzpatrick_skin_type: str = Form(...),
    body_area: str = Form(...),
    prior_treatments: str = Form(...),
    baseline_severity: str = Form(...),
) -> PredictResponse:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Upload must be a JPEG or PNG image.")
    if not (0 < age < 130):
        raise HTTPException(status_code=422, detail="Age must be between 1 and 129.")
    for value, label in [
        (sex, "sex"),
        (race_ethnicity, "race_ethnicity"),
        (fitzpatrick_skin_type, "fitzpatrick_skin_type"),
        (body_area, "body_area"),
        (prior_treatments, "prior_treatments"),
        (baseline_severity, "baseline_severity"),
    ]:
        if not value.strip():
            raise HTTPException(status_code=422, detail=f"{label} is required.")

    return build_mock_predict_response()
