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
    ExactMatch,
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
    image_histogram,
    image_signature,
    signature_similarity,
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

# Perceptual-signature similarity above which an uploaded photo is treated as the
# *same* image as a reference before-photo (our matching-algorithm probe). We take
# the max of a precise structural signature (great for identical files) and a
# crop/translation-tolerant color histogram (great for screenshots/re-crops of a
# reference photo). Different patients top out well below this (structural <=0.60,
# histogram <=0.86), so 0.90 keeps a safe margin against false positives.
EXACT_MATCH_THRESHOLD = 0.90
# Confidence floor applied to the matched biologic when an exact image match is found.
EXACT_MATCH_LIKELIHOOD_FLOOR = 96

DISCLAIMER = (
    "DermaMatch is prototype decision-support for discussion with a dermatologist. "
    "It is not a diagnosis, prescription, or medical advice."
)
PRIVACY_NOTICE = (
    "Uploaded images are analyzed for this session only and are not stored as an "
    "account or EHR record."
)

# Lifestyle / side-effect-profile heuristics derived from the optional "typical day"
# note. These reflect *dosing-convenience and tolerability considerations* a patient
# might raise with their dermatologist — not efficacy claims. Each rule may nudge the
# recommendation toward the biologic whose profile better fits that lifestyle, and/or
# surface a plain-language consideration. Dupixent maintenance is ~every 2 weeks;
# Ebglyss maintenance is ~every 4 weeks after loading.
LIFESTYLE_NUDGE_POINTS = 3
LIFESTYLE_NUDGE_CAP = 6
LIFESTYLE_RULES: Tuple[Tuple[Tuple[str, ...], Optional[str], str], ...] = (
    (
        (
            "travel", "traveling", "on the road", "flight", "flying", "frequent flyer",
            "busy", "hectic", "packed schedule", "no time", "always moving", "on the go",
            "commute", "commuting", "shift work", "long hours", "overtime",
        ),
        "Ebglyss",
        "Your day sounds busy and on-the-go — Ebglyss's less frequent maintenance dosing "
        "(about every 4 weeks vs every 2 weeks) may be easier to keep up with.",
    ),
    (
        (
            "needle", "needles", "injection", "injections", "shot", "shots",
            "afraid of needles", "hate shots", "phobia", "squeamish",
        ),
        "Ebglyss",
        "If frequent injections are a concern, Ebglyss's less frequent maintenance "
        "schedule means fewer shots to manage.",
    ),
    (
        (
            "child", "children", "kids", "toddler", "baby", "parent", "parenting",
            "caregiver", "caring for", "family to look after",
        ),
        "Ebglyss",
        "Caregiving leaves little time for clinic visits — a less frequent dosing "
        "schedule (Ebglyss) may fit a full household routine better.",
    ),
    (
        (
            "screen", "screens", "computer", "monitor", "coding", "reading",
            "contact lens", "contact lenses", "contacts", "dry eye", "dry eyes",
            "eye strain", "gaming",
        ),
        None,
        "You spend meaningful time on screens or wear contacts — discuss conjunctivitis "
        "(eye irritation) monitoring with your dermatologist, as it is a known side "
        "effect of both biologics.",
    ),
    (
        (
            "outdoor", "outdoors", "outside", "sun", "sweat", "sweating", "gym",
            "run", "running", "jog", "sport", "sports", "athlete", "active",
            "hiking", "hike", "swim", "swimming", "cycling", "bike",
        ),
        None,
        "An active, outdoor routine can affect skin barrier and irritation — worth "
        "raising when you discuss day-to-day tolerability with your dermatologist.",
    ),
)


def _analyze_lifestyle(daily_routine: str) -> Tuple[Dict[str, int], List[str]]:
    """Map the free-text 'typical day' to biologic nudges + plain considerations."""
    nudges: Dict[str, int] = {biologic: 0 for biologic in BIOLOGICS}
    considerations: List[str] = []
    text = (daily_routine or "").lower().strip()
    if not text:
        return nudges, considerations
    for keywords, biologic, note in LIFESTYLE_RULES:
        if any(keyword in text for keyword in keywords):
            considerations.append(note)
            if biologic in nudges:
                nudges[biologic] = min(
                    LIFESTYLE_NUDGE_CAP, nudges[biologic] + LIFESTYLE_NUDGE_POINTS
                )
    return nudges, considerations


def _apply_lifestyle_nudges(
    likelihoods: List[BiologicLikelihood], nudges: Dict[str, int]
) -> List[BiologicLikelihood]:
    if not any(nudges.values()):
        return likelihoods
    adjusted: List[BiologicLikelihood] = []
    for item in likelihoods:
        delta = nudges.get(item.biologic, 0)
        if delta and item.likelihood_pct > 0:
            new_pct = round(min(95, max(0, item.likelihood_pct + delta)), 1)
            adjusted.append(
                item.model_copy(
                    update={
                        "likelihood_pct": new_pct,
                        "caveat": (
                            f"{item.caveat} Adjusted for lifestyle fit "
                            f"(+{delta} for dosing-convenience considerations)."
                        ),
                    }
                )
            )
        else:
            adjusted.append(item)
    return adjusted


@dataclass(frozen=True)
class _ScoredCase:
    case: ImageReferenceCase
    similarity: float
    distances: Dict[str, float]


def _normalized(features: PatientFeatures, field: str) -> float:
    value = float(getattr(features, field))
    return value / 100.0 if field.endswith("_pct") else value


def _age_bounds(cases: List[ImageReferenceCase]) -> Optional[Tuple[int, int]]:
    ages = [case.age for case in cases if case.age is not None]
    if not ages:
        return None
    return min(ages), max(ages)


def _age_norm(age: int, low: int, high: int) -> float:
    if high <= low:
        return 0.5
    return min(1.0, max(0.0, (age - low) / (high - low)))


def _feature_distances(
    patient: PatientFeatures,
    patient_age_norm: float,
    case: ImageReferenceCase,
    age_bounds: Optional[Tuple[int, int]],
) -> Dict[str, float]:
    distances: Dict[str, float] = {}
    for field in BIOMARKER_FIELDS:
        distances[field] = abs(
            _normalized(patient, field) - _normalized(case.before_features, field)
        )
    # Age only contributes when both the patient and the case have a known age.
    if case.age is not None and age_bounds is not None:
        low, high = age_bounds
        distances["age"] = abs(patient_age_norm - _age_norm(case.age, low, high))
    return distances


def _weighted_similarity(distances: Dict[str, float]) -> float:
    used_weight = sum(MATCH_WEIGHTS[field] for field in distances if field in MATCH_WEIGHTS)
    if used_weight <= 0:
        return 0.0
    total = sum(
        MATCH_WEIGHTS[field] * (distances[field] ** 2)
        for field in distances
        if field in MATCH_WEIGHTS
    )
    distance = math.sqrt(total / used_weight)  # 0..1, normalized over used features
    return float(min(1.0, max(0.0, 1.0 - distance)))


def _score_cases(
    patient: PatientFeatures,
    patient_age: int,
    cases: List[ImageReferenceCase],
) -> List[_ScoredCase]:
    age_bounds = _age_bounds(cases)
    patient_age_norm = (
        _age_norm(patient_age, age_bounds[0], age_bounds[1]) if age_bounds else 0.5
    )
    scored: List[_ScoredCase] = []
    for case in cases:
        distances = _feature_distances(patient, patient_age_norm, case, age_bounds)
        scored.append(
            _ScoredCase(
                case=case,
                similarity=_weighted_similarity(distances),
                distances=distances,
            )
        )
    scored.sort(key=lambda item: item.similarity, reverse=True)
    return scored


def _confidence_label(best_similarity: float, count: int) -> str:
    if count == 0:
        return "no matching cases"
    if best_similarity >= 0.9:
        return "high match confidence"
    if best_similarity >= 0.78:
        return "moderate match confidence"
    return "low match confidence (small reference set)"


# The single closest case dominates so an (near-)exact photo match recommends the
# biologic that case actually cleared on; remaining neighbors nuance the score.
BEST_MATCH_WEIGHT = 0.7
MEAN_MATCH_WEIGHT = 0.3


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
            caveat=f"No {biologic} success cases in the reference dataset yet.",
        )

    best_similarity = matches[0].similarity
    mean_similarity = sum(item.similarity for item in matches) / len(matches)
    match_score = BEST_MATCH_WEIGHT * best_similarity + MEAN_MATCH_WEIGHT * mean_similarity
    return BiologicLikelihood(
        biologic=biologic,
        likelihood_pct=round(match_score * 100, 1),
        confidence_label=_confidence_label(best_similarity, len(matches)),
        matched_case_count=len(matches),
        weighted_outcome_score=round(match_score, 3),
        caveat=(
            f"Based on how closely your skin matches {len(matches)} patient(s) who "
            f"improved on {biologic}; every reference case is a success case."
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
                demographic_summary=(
                    f"Age {case.age}" if case.age is not None else "Age not recorded"
                ),
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


def _find_exact_match(
    image_bytes: bytes, cases: List[ImageReferenceCase]
) -> Optional[Tuple[ImageReferenceCase, float]]:
    """Return the reference case whose before-photo is (near-)identical, if any.

    Robust to screenshots/re-crops: combines a precise structural signature with a
    crop/translation-tolerant color histogram and keeps whichever is stronger.
    """
    upload_signature = image_signature(image_bytes)
    upload_histogram = image_histogram(image_bytes)
    best: Optional[Tuple[ImageReferenceCase, float]] = None
    for case in cases:
        structural = signature_similarity(upload_signature, case.before_signature)
        histogram = signature_similarity(upload_histogram, case.before_histogram)
        similarity = max(structural, histogram)
        if similarity >= EXACT_MATCH_THRESHOLD and (best is None or similarity > best[1]):
            best = (case, similarity)
    return best


def _explanation(
    patient: PatientFeatures,
    likelihoods: List[BiologicLikelihood],
    total_cases: int,
    exact: Optional[ExactMatch],
    lifestyle_considerations: Optional[List[str]] = None,
) -> Explanation:
    ranked = sorted(likelihoods, key=lambda item: item.likelihood_pct, reverse=True)
    best = ranked[0]
    top_biomarkers = _top_biomarkers(patient)
    lead_biomarker = top_biomarkers[0].label.lower() if top_biomarkers else "visual biomarkers"

    if exact is not None:
        summary = (
            f"Exact image match found: your photo is visually identical to reference "
            f"case {exact.case_id}, who cleared on {exact.biologic}. We are therefore "
            f"highly confident {exact.biologic} is the closest-matched treatment for you."
        )
        rationale = (
            f"{exact.biologic} is recommended because your uploaded photo is a near-exact "
            f"match to reference case {exact.case_id}, a patient who improved on "
            f"{exact.biologic} — the strongest possible image-grounded signal."
        )
    else:
        summary = (
            f"Your photo's visual biomarkers were matched against {total_cases} success "
            f"cases. {best.biologic} is the closest match — your skin most resembles "
            f"patients who improved on {best.biologic}."
        )
        rationale = None
        if len(ranked) > 1:
            other = ranked[1]
            margin = round(best.likelihood_pct - other.likelihood_pct, 1)
            margin_text = f"{margin:g}-point" if margin > 0 else "narrow"
            rationale = (
                f"{best.biologic} edges out {other.biologic} ({best.likelihood_pct:g}% vs "
                f"{other.likelihood_pct:g}%, a {margin_text} margin) because your closest "
                f"reference matches — driven mainly by your {lead_biomarker} — are patients "
                f"who cleared on {best.biologic} rather than {other.biologic}."
            )

    return Explanation(
        summary=summary,
        recommendation_rationale=rationale,
        top_contributing_biomarkers=top_biomarkers,
        lifestyle_considerations=lifestyle_considerations or [],
    )


def build_predict_response(
    image_bytes: bytes,
    age: int,
    repository: ImageReferenceRepository,
    daily_routine: str = "",
) -> PredictResponse:
    cases = repository.list_cases()
    patient_features, quality = extract_biomarkers(image_bytes)
    scored = _score_cases(patient_features, age, cases)

    likelihoods = [_likelihood_for_biologic(biologic, scored) for biologic in BIOLOGICS]
    matched_patients = _matched_patients(scored)

    lifestyle_nudges, lifestyle_considerations = _analyze_lifestyle(daily_routine)

    exact: Optional[ExactMatch] = None
    exact_hit = _find_exact_match(image_bytes, cases)
    if exact_hit is not None:
        case, similarity = exact_hit
        exact = ExactMatch(
            case_id=case.case_id,
            biologic=case.biologic,
            similarity=round(similarity, 4),
            before_image_url=_media_url(case.before_path),
            after_image_url=_media_url(case.after_path),
        )
        # Boost the matched biologic to a high-confidence recommendation.
        likelihoods = [
            item.model_copy(
                update={
                    "likelihood_pct": max(
                        item.likelihood_pct, EXACT_MATCH_LIKELIHOOD_FLOOR
                    ),
                    "confidence_label": "very high confidence",
                    "caveat": (
                        f"Your uploaded photo is visually identical to {case.case_id}, "
                        f"who improved on {case.biologic}."
                    ),
                }
            )
            if item.biologic == case.biologic
            else item
            for item in likelihoods
        ]
    else:
        # Lifestyle only nudges the recommendation when there is no exact image match
        # (an exact match is a far stronger, image-grounded signal).
        likelihoods = _apply_lifestyle_nudges(likelihoods, lifestyle_nudges)

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
        explanation=_explanation(
            patient_features,
            likelihoods,
            len(cases),
            exact,
            lifestyle_considerations,
        ),
        heatmap=Heatmap(
            overlay_url=None,
            legend="Visual biomarker overlay is not yet rendered in this prototype.",
        ),
        matched_patients=matched_patients,
        warnings=warnings,
        exact_match=exact,
    )
