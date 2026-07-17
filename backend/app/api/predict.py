from functools import lru_cache

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.predict import PredictResponse
from app.services.image_dataset import ImageDatasetError, ImageReferenceRepository
from app.services.image_predict import build_predict_response
from app.services.preprocessing import InvalidImageError

router = APIRouter(prefix="/api", tags=["predict"])


@lru_cache(maxsize=1)
def get_repository() -> ImageReferenceRepository:
    """Cache the reference repository so images are only processed once."""
    return ImageReferenceRepository()


@router.post("/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    race_ethnicity: str = Form(...),
    body_area: str = Form(...),
    eczema_duration: str = Form(...),
    itch_severity: str = Form(...),
    atopic_comorbidities: str = Form(...),
    tried_biologics: str = Form(...),
    biologics_stopped_reason: str = Form(""),
    nonbiologic_treatments: str = Form(...),
    daily_routine: str = Form(...),
) -> PredictResponse:
    if image.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Upload must be a JPEG or PNG image.")
    if not (0 < age < 130):
        raise HTTPException(status_code=422, detail="Age must be between 1 and 129.")
    for value, label in [
        (sex, "sex"),
        (race_ethnicity, "race_ethnicity"),
        (body_area, "body_area"),
        (eczema_duration, "eczema_duration"),
        (itch_severity, "itch_severity"),
        (atopic_comorbidities, "atopic_comorbidities"),
        (tried_biologics, "tried_biologics"),
        (nonbiologic_treatments, "nonbiologic_treatments"),
        (daily_routine, "daily_routine"),
    ]:
        if not value.strip():
            raise HTTPException(status_code=422, detail=f"{label} is required.")
    if tried_biologics.strip().lower() == "yes" and not biologics_stopped_reason.strip():
        raise HTTPException(
            status_code=422,
            detail="biologics_stopped_reason is required when biologics were tried.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    try:
        repository = get_repository()
        return build_predict_response(
            image_bytes,
            age,
            repository,
            daily_routine=daily_routine,
            atopic_comorbidities=atopic_comorbidities,
        )
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"That image could not be analyzed: {exc}",
        ) from exc
    except ImageDatasetError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Reference dataset is unavailable: {exc}",
        ) from exc
