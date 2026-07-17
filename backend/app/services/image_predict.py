"""Image-driven similarity matching for the 4-column reference dataset.

A patient's uploaded photo is reduced to a biomarker vector (plus age) and matched
against reference *Before* images. For each biologic, the expected response is the
distance-weighted average of the improvement scores of the most similar cases.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.schemas.predict import (
    BiologicLikelihood,
    ContributingBiomarker,
    Explanation,
    Heatmap,
    MatchedPatient,
    PatientFeatures,
    PredictResponse,
)
from app.services.biomarker_extraction import extract_biomarkers
from app.services.image_dataset import (
    ImageReferenceCase,
    ImageReferenceRepository,
)

BIOLOGICS = ("Dupixent", "Ebglyss")
REFERENCE_MEDIA_PREFIX = "/api/reference-media"

# Feature weights for patient <-> reference similarity (sum ~= 1.0).
MATCH_WEIGHTS: Dict[str, float] = {
    "inflammation_score": 0.22,
    "erythema_score": 0.20,
    "lesion_coverage_pct": 0.18,
    "texture_score": 0.12,
    "affected_body_area_pct": 0.10,
    "dryness_scaling_score": 0.10,
    "age": 0.08,
}
BIOMARKER_FIELDS = (
    "erythema_score",
    "lesion_coverage_pct",
    "texture_score",
    "dryness_scaling_score",
    "inflammation_score",
    "affected_body_area_pct",
)
BIOMARKER_LABELS: Dict[str, str] = {
    "erythema_score": "redness intensity",
    "lesion_coverage_pct": "lesion coverage",
    "texture_score": "skin texture",
    "dryness_scaling_score": "dryness / scaling",
    "inflammation_score": "overall inflammation",
    "affected_body_area_pct": "affected area",
    "age": "age",
}

TOP_K_NEIGHBORS = 5
MAX_MATCHED_PATIENTS = 3
MAX_REASONS = 3
# Distance below which two features count as an aligned "matching reason".
REASON_DISTANCE_CEILING = 0.25

DISCLAIMER = (
    "DermaMatch is prototype decision-support for discussion with a dermatologist. "
    "It is not a diagnosis, prescription, or medical advice."
)
PRIVACY_NOTICE = (
    "Uploaded images are analyzed for this session only and are not stored as an "
    "account or EHR record."
)


@dataclass(frozen=True)
class _ScoredCase:
    case: ImageReferenceCase
    similarity: float
    distances: Dict[str, float]


def _normalized(features: PatientFeatures, field: str) -> float:
    value = float(getattr(features, field))
    return value / 100.0 if field.endswith("_pct") else value


def _age_bounds(cases: List[ImageReferenceCase]) -> Tuple[int, int]:
    ages = [case.age for case in cases]
    return min(ages), max(ages)


def _age_norm(age: int, low: int, high: int) -> float:
    if high <= low:
        return 0.5
    return min(1.0, max(0.0, (age - low) / (high - low)))


def _feature_distances(
    patient: PatientFeatures,
    patient_age_norm: float,
    case: ImageReferenceCase,
    age_low: int,
    age_high: int,
) -> Dict[str, float]:
    distances: Dict[str, float] = {}
    for field in BIOMARKER_FIELDS:
        distances[field] = abs(
            _normalized(patient, field) - _normalized(case.before_features, field)
        )
    distances["age"] = abs(patient_age_norm - _age_norm(case.age, age_low, age_high))
    return distances


def _weighted_similarity(distances: Dict[str, float]) -> float:
    total = sum(
        MATCH_WEIGHTS[field] * (distances[field] ** 2) for field in MATCH_WEIGHTS
    )
    distance = math.sqrt(total)  # 0..1 given normalized features and sum(weights)~1
    return float(min(1.0, max(0.0, 1.0 - distance)))


def _score_cases(
    patient: PatientFeatures,
    patient_age: int,
    cases: List[ImageReferenceCase],
) -> List[_ScoredCase]:
    age_low, age_high = _age_bounds(cases)
    patient_age_norm = _age_norm(patient_age, age_low, age_high)
    scored: List[_ScoredCase] = []
    for case in cases:
        distances = _feature_distances(
            patient, patient_age_norm, case, age_low, age_high
        )
        scored.append(
            _ScoredCase(
                case=case,
                similarity=_weighted_similarity(distances),
                distances=distances,
            )
        )
    scored.sort(key=lambda item: item.similarity, reverse=True)
    return scored


def _confidence_label(count: int, avg_similarity: float) -> str:
    if count == 0:
        return "no matching cases"
    if count >= 4 and avg_similarity >= 0.75:
        return "moderate confidence (small reference set)"
    if count >= 2:
        return "low confidence (small reference set)"
    return "very low confidence (single case)"


def _likelihood_for_biologic(
    biologic: str, scored: List[_ScoredCase]
) -> BiologicLikelihood:
    matches = [item for item in scored if item.case.biologic == biologic][
        :TOP_K_NEIGHBORS
    ]
    if not matches:
        return BiologicLikelihood(
            biologic=biologic,
            likelihood_pct=0,
            confidence_label="no matching cases",
            matched_case_count=0,
            weighted_outcome_score=0.0,
            caveat=f"No {biologic} cases in the reference dataset yet.",
        )

    weight_sum = sum(max(item.similarity, 1e-6) for item in matches)
    weighted_outcome = (
        sum(max(item.similarity, 1e-6) * item.case.improvement_score for item in matches)
        / weight_sum
    )
    avg_similarity = sum(item.similarity for item in matches) / len(matches)
    return BiologicLikelihood(
        biologic=biologic,
        likelihood_pct=int(round(weighted_outcome * 100)),
        confidence_label=_confidence_label(len(matches), avg_similarity),
        matched_case_count=len(matches),
        weighted_outcome_score=round(weighted_outcome, 3),
        caveat=(
            "Estimated from the Before/After improvement of the most similar "
            f"{len(matches)} reference case(s); small samples are uncertain."
        ),
    )


def _matching_reasons(scored: _ScoredCase) -> List[str]:
    aligned = sorted(
        (
            (field, distance)
            for field, distance in scored.distances.items()
            if distance <= REASON_DISTANCE_CEILING
        ),
        key=lambda item: item[1],
    )
    reasons = [
        f"Similar {BIOMARKER_LABELS.get(field, field)}" for field, _ in aligned[:MAX_REASONS]
    ]
    if not reasons:
        reasons.append("Closest overall biomarker profile in the dataset")
    return reasons


def _media_url(relative_path: str) -> str:
    return f"{REFERENCE_MEDIA_PREFIX}/{relative_path.lstrip('/')}"


def _matched_patients(scored: List[_ScoredCase]) -> List[MatchedPatient]:
    matched: List[MatchedPatient] = []
    for item in scored[:MAX_MATCHED_PATIENTS]:
        case = item.case
        matched.append(
            MatchedPatient(
                case_id=case.case_id,
                similarity=round(item.similarity, 3),
                biologic_used=case.biologic,
                outcome_label=case.outcome_label,
                outcome_score=round(case.improvement_score, 3),
                demographic_summary=f"Age {case.age}",
                matching_reasons=_matching_reasons(item),
                before_image_url=_media_url(case.before_path),
                after_image_url=_media_url(case.after_path),
            )
        )
    return matched


def _top_biomarkers(patient: PatientFeatures) -> List[ContributingBiomarker]:
    ranked = sorted(
        BIOMARKER_FIELDS,
        key=lambda field: _normalized(patient, field),
        reverse=True,
    )
    contributors: List[ContributingBiomarker] = []
    for field in ranked[:3]:
        contributors.append(
            ContributingBiomarker(
                name=field,
                label=BIOMARKER_LABELS.get(field, field),
                patient_value=round(float(getattr(patient, field)), 3),
                direction="drives the similarity match",
                weight=round(MATCH_WEIGHTS.get(field, 0.0), 2),
            )
        )
    return contributors


def _explanation(
    patient: PatientFeatures,
    likelihoods: List[BiologicLikelihood],
    total_cases: int,
) -> Explanation:
    best = max(likelihoods, key=lambda item: item.likelihood_pct)
    summary = (
        f"Your photo's strongest visual biomarkers were matched against {total_cases} "
        f"reference case(s). {best.biologic} shows the highest estimated response "
        f"({best.likelihood_pct}%), based on the Before/After improvement of the most "
        "similar patients."
    )
    return Explanation(
        summary=summary,
        top_contributing_biomarkers=_top_biomarkers(patient),
    )


def build_predict_response(
    image_bytes: bytes,
    age: int,
    repository: ImageReferenceRepository,
) -> PredictResponse:
    cases = repository.list_cases()
    patient_features, quality = extract_biomarkers(image_bytes)
    scored = _score_cases(patient_features, age, cases)

    likelihoods = [_likelihood_for_biologic(biologic, scored) for biologic in BIOLOGICS]
    matched_patients = _matched_patients(scored)

    warnings: List[str] = list(quality.warnings)
    if len(cases) < 10:
        warnings.append("small_reference_dataset")

    return PredictResponse(
        request_id=f"dm-{uuid.uuid4().hex[:8]}",
        mock=False,
        disclaimer=DISCLAIMER,
        privacy_notice=PRIVACY_NOTICE,
        patient_features=patient_features,
        likelihoods=likelihoods,
        explanation=_explanation(patient_features, likelihoods, len(cases)),
        heatmap=Heatmap(
            overlay_url=None,
            legend="Visual biomarker overlay is not yet rendered in this prototype.",
        ),
        matched_patients=matched_patients,
        warnings=warnings,
    )
