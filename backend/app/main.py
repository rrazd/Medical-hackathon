from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.api.predict import router as predict_router
from app.config import get_settings
from app.services.image_dataset import DEFAULT_DATA_ROOT


class HealthResponse(BaseModel):
    ok: bool


app = FastAPI(title="DermaMatch API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(ok=True)


app.include_router(predict_router)

# Serve reference before/after images referenced by matched cases.
if DEFAULT_DATA_ROOT.is_dir():
    app.mount(
        "/api/reference-media",
        StaticFiles(directory=str(DEFAULT_DATA_ROOT)),
        name="reference-media",
    )
