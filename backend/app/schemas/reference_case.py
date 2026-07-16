from pathlib import Path
from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.predict import PatientFeatures


BIOMARKER_FIELDS = tuple(PatientFeatures.model_fields.keys())


class ReferenceCaseRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(..., pattern=r"^DM-\d{3}$")
    biologic: Literal["Dupixent", "Ebglyss"]
    outcome_label: Literal["responder", "partial_responder", "non_responder"]
    outcome_score: float = Field(..., ge=0.0, le=1.0)
    age: int = Field(..., ge=1, le=129)
    sex: Literal["female", "male", "nonbinary", "prefer_not_to_say"]
    race_ethnicity: Literal[
        "white",
        "asian",
        "hispanic_latino",
        "black",
        "multiracial",
        "middle_eastern_north_african",
        "native_hawaiian_pacific_islander",
        "other",
        "prefer_not_to_say",
    ]
    fitzpatrick_skin_type: Literal["I", "II", "III", "IV", "V", "VI"]
    body_area: Literal["arm", "flexural", "hand", "leg", "trunk", "face", "neck", "scalp"]
    baseline_severity: Literal["mild", "moderate", "severe"]
    before_image_path: str
    after_image_path: str
    erythema_score: float = Field(..., ge=0.0, le=1.0)
    lesion_coverage_pct: float = Field(..., ge=0.0, le=100.0)
    texture_score: float = Field(..., ge=0.0, le=1.0)
    dryness_scaling_score: float = Field(..., ge=0.0, le=1.0)
    inflammation_score: float = Field(..., ge=0.0, le=1.0)
    affected_body_area_pct: float = Field(..., ge=0.0, le=100.0)
    followup_weeks: int = Field(..., ge=4, le=52)

    @field_validator("*", mode="before")
    @classmethod
    def normalize_blank_strings(cls, value):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            return stripped
        return value

    @field_validator("before_image_path", "after_image_path")
    @classmethod
    def image_paths_are_present(cls, value: str) -> str:
        if not value:
            raise ValueError("image path is required")
        return value

    @model_validator(mode="after")
    def validate_outcome_consistency(self):
        expected_scores: Dict[str, float] = {
            "responder": 1.0,
            "partial_responder": 0.5,
            "non_responder": 0.0,
        }
        expected = expected_scores[self.outcome_label]
        if self.outcome_score != expected:
            raise ValueError(
                f"outcome_score must be {expected} for outcome_label {self.outcome_label}"
            )
        return self

    def biomarker_vector(self) -> List[float]:
        return [float(getattr(self, field_name)) for field_name in BIOMARKER_FIELDS]


class ReferenceCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    row: ReferenceCaseRow
    before_image_file: Path
    after_image_file: Path
