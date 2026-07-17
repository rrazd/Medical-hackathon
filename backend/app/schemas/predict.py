from typing import List, Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    erythema_score: float
    lesion_coverage_pct: float
    texture_score: float
    dryness_scaling_score: float
    inflammation_score: float
    affected_body_area_pct: float


class BiologicLikelihood(BaseModel):
    biologic: str
    likelihood_pct: float
    confidence_label: str
    matched_case_count: int
    weighted_outcome_score: float
    caveat: str


class ContributingBiomarker(BaseModel):
    name: str
    label: str
    patient_value: float
    direction: str
    weight: float


class Explanation(BaseModel):
    summary: str
    recommendation_rationale: Optional[str] = None
    top_contributing_biomarkers: List[ContributingBiomarker]
    lifestyle_considerations: List[str] = Field(default_factory=list)


class Heatmap(BaseModel):
    overlay_url: Optional[str] = None
    legend: str


class MatchedPatient(BaseModel):
    case_id: str
    similarity: float
    biologic_used: str
    outcome_label: str
    outcome_score: float
    demographic_summary: str
    matching_reasons: List[str]
    before_image_url: Optional[str] = None
    after_image_url: Optional[str] = None


class ExactMatch(BaseModel):
    case_id: str
    biologic: str
    similarity: float
    before_image_url: Optional[str] = None
    after_image_url: Optional[str] = None


class SeverityScores(BaseModel):
    """Photo-estimated baseline atopic-dermatitis severity indices.

    Derived from the uploaded photo's visual biomarkers as prototype proxies for
    two standard clinical measures. Not a substitute for a clinician's scoring.
    """

    iga: int  # Investigator's Global Assessment, 0 (clear) – 4 (severe)
    iga_label: str
    easi: float  # Eczema Area and Severity Index, 0 – 72
    easi_max: float = 72.0
    severity_label: str


class PredictResponse(BaseModel):
    request_id: str = Field(..., examples=["mock-001"])
    mock: bool
    disclaimer: str
    privacy_notice: str
    patient_features: PatientFeatures
    severity: SeverityScores
    likelihoods: List[BiologicLikelihood]
    explanation: Explanation
    heatmap: Heatmap
    matched_patients: List[MatchedPatient]
    warnings: List[str]
    exact_match: Optional[ExactMatch] = None
