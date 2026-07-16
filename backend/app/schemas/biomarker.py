from typing import List, Optional

from pydantic import BaseModel, Field


class QualityReport(BaseModel):
    warnings: List[str] = Field(default_factory=list)
    blur_laplacian_var: float
    exposure_mean: float
    underexposed_pct: float
    overexposed_pct: float
    color_cast_score: float
    skin_pixel_fraction: float
    estimated_skin_tone_ita: Optional[float] = None
    segmentation_confidence: float
