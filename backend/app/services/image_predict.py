"""Image-driven similarity matching for the 4-column reference dataset.

A patient's uploaded photo is reduced to a biomarker vector (plus age) and matched
against reference *Before* images. For each biologic, the expected response is the
distance-weighted average of the improvement scores of the most similar cases.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from pathlib import Path
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
    SeverityScores,
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
                            f"{item.caveat} Adjusted for lifestyle and comorbidity "
                            f"fit (+{delta})."
                        ),
                    }
                )
            )
        else:
            adjusted.append(item)
    return adjusted


# Dupixent (dupilumab) is FDA-approved not only for atopic dermatitis but also for
# asthma and chronic rhinosinusitis (which underlies allergic-rhinitis / "hay fever"
# type disease). Ebglyss (lebrikizumab) is currently approved for atopic dermatitis
# only. A patient with these atopic comorbidities may therefore favor Dupixent, since
# a single biologic could address more than their skin. This is a treatment-selection
# consideration to raise with a clinician — not an efficacy guarantee.
COMORBIDITY_NUDGE_POINTS = 6

_COMORBIDITY_LABELS: Dict[str, str] = {
    "asthma": "asthma",
    "hay-fever": "hay fever (allergic rhinitis)",
    "both": "asthma and hay fever",
}


def _analyze_comorbidities(
    atopic_comorbidities: str,
) -> Tuple[Dict[str, int], List[str]]:
    """Map reported asthma / hay-fever comorbidity to a Dupixent nudge + note."""
    nudges: Dict[str, int] = {biologic: 0 for biologic in BIOLOGICS}
    considerations: List[str] = []
    value = (atopic_comorbidities or "").strip().lower()
    if value not in _COMORBIDITY_LABELS:
        return nudges, considerations
    nudges["Dupixent"] = COMORBIDITY_NUDGE_POINTS
    label = _COMORBIDITY_LABELS[value]
    considerations.append(
        f"You reported {label}. Dupixent is also FDA-approved for these atopic "
        f"conditions, so one biologic could treat more than your skin — worth raising "
        f"with your dermatologist. Ebglyss is currently approved for eczema only."
    )
    return nudges, considerations


def _merge_nudges(*nudge_maps: Dict[str, int]) -> Dict[str, int]:
    merged: Dict[str, int] = {biologic: 0 for biologic in BIOLOGICS}
    for nudge_map in nudge_maps:
        for biologic, delta in nudge_map.items():
            merged[biologic] = merged.get(biologic, 0) + delta
    return merged


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


# The recommendation is grounded in the single most similar reference patient for
# each biologic: "how closely does your skin match the closest patient who cleared
# on this drug?". Averaging over several neighbors was tried but biased the result
# toward whichever biologic has the tighter reference cluster (a higher neighbor
# mean) regardless of the patient — so a density-invariant nearest-neighbor score
# is used instead, letting the patient's own features decide the winner.
#
# Even nearest-neighbor similarity is not directly comparable between the two
# cohorts: over the distribution of real uploads one biologic scores higher purely
# as a dataset artifact. A small per-biologic calibration offset (precomputed by
# scripts/build_calibration.py over a large sample of diverse synthetic photos, and
# stored in data/calibration.json) shifts both biologics to a shared mean so a
# typical upload scores them equally and the patient's own biomarkers decide.
_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "calibration.json"
)


def _load_calibration_offsets() -> Dict[str, float]:
    try:
        payload = json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8"))
        offsets = payload.get("offsets", {})
        return {biologic: float(offsets.get(biologic, 0.0)) for biologic in BIOLOGICS}
    except (OSError, ValueError, TypeError):
        # No calibration artifact (e.g. tests / fresh dataset): fall back to raw
        # nearest-neighbor scoring rather than failing.
        return {biologic: 0.0 for biologic in BIOLOGICS}


_CALIBRATION_OFFSETS = _load_calibration_offsets()


def _likelihood_for_biologic(
    biologic: str,
    scored: List[_ScoredCase],
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
    # Cap below 1.0 so no result reads as a literal "100% likelihood of improvement"
    # (medical-safety: the tool is decision-support, not a guarantee).
    match_score = min(
        0.99, max(0.0, best_similarity + _CALIBRATION_OFFSETS.get(biologic, 0.0))
    )
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


def _biomarker_differentiators(
    recommended: str, other: str, scored: List[_ScoredCase]
) -> List[Tuple[str, float]]:
    """Rank biomarkers by how much they pull the match toward `recommended` vs `other`.

    Returns (field, weighted_pull) sorted by descending pull. A positive value means
    the patient's skin on that feature looks more like the recommended cohort (a driver
    of the pick); negative means it leans toward the other biologic.
    """

    def top_matches(biologic: str) -> List[_ScoredCase]:
        return [m for m in scored if m.case.biologic == biologic][:TOP_K_NEIGHBORS]

    rec_matches = top_matches(recommended)
    oth_matches = top_matches(other)
    if not rec_matches or not oth_matches:
        return []

    ranked: List[Tuple[str, float]] = []
    for field in BIOMARKER_FIELDS:
        rec_d = [m.distances[field] for m in rec_matches if field in m.distances]
        oth_d = [m.distances[field] for m in oth_matches if field in m.distances]
        if not rec_d or not oth_d:
            continue
        rec_mean = sum(rec_d) / len(rec_d)
        oth_mean = sum(oth_d) / len(oth_d)
        weighted_pull = (oth_mean - rec_mean) * MATCH_WEIGHTS.get(field, 0.0)
        ranked.append((field, weighted_pull))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked


def _cohort_feature_means(biologic: str, scored: List[_ScoredCase]) -> Dict[str, float]:
    """Average normalized (0–1) biomarker values of a biologic's closest reference cases."""
    matches = [m for m in scored if m.case.biologic == biologic][:TOP_K_NEIGHBORS]
    means: Dict[str, float] = {}
    for field in BIOMARKER_FIELDS:
        values = [_normalized(m.case.before_features, field) for m in matches]
        if values:
            means[field] = sum(values) / len(values)
    return means


def _level_word(value: float) -> str:
    if value < 0.2:
        return "very low"
    if value < 0.4:
        return "low"
    if value < 0.6:
        return "moderate"
    if value < 0.8:
        return "high"
    return "very high"


def _recommendation_rationale(
    patient: PatientFeatures,
    best: BiologicLikelihood,
    other: BiologicLikelihood,
    scored: List[_ScoredCase],
    comorbidity_label: Optional[str],
) -> str:
    diffs = _biomarker_differentiators(best.biologic, other.biologic, scored)
    drivers = [field for field, val in diffs if val > 1e-4]
    against = [field for field, val in diffs if val < -1e-4]
    net = sum(val for _, val in diffs)
    margin = round(best.likelihood_pct - other.likelihood_pct, 1)

    # Comorbidity was the deciding factor: biomarkers alone don't favor the pick.
    if comorbidity_label and net <= 1e-4:
        return (
            f"On skin biomarkers alone, your matches for {best.biologic} and "
            f"{other.biologic} are nearly even. Because you reported {comorbidity_label} "
            f"— which {best.biologic} is also approved to treat — it becomes the better "
            f"overall fit for you."
        )

    if not drivers:
        return (
            f"{best.biologic} and {other.biologic} are almost tied ("
            f"{best.likelihood_pct:g}% vs {other.likelihood_pct:g}%); your skin resembles "
            f"successful patients from both groups about equally, so this is a close call "
            f"to discuss with your dermatologist."
        )

    rec_means = _cohort_feature_means(best.biologic, scored)
    oth_means = _cohort_feature_means(other.biologic, scored)

    def clause(field: str) -> str:
        label = BIOMARKER_LABELS.get(field, field)
        pv = _normalized(patient, field)
        rec = rec_means.get(field, pv)
        oth = oth_means.get(field, pv)
        return (
            f"your {label} is {_level_word(pv)} (~{round(pv * 100)}/100), which lines up "
            f"with the {best.biologic} responders (~{round(rec * 100)}) and sits apart from "
            f"the {other.biologic} responders (~{round(oth * 100)})"
        )

    driver_clauses = "; ".join(clause(field) for field in drivers[:2])
    sentence = f"Specifically, {driver_clauses}."

    if against:
        field = against[0]
        label = BIOMARKER_LABELS.get(field, field)
        pv = _normalized(patient, field)
        oth = oth_means.get(field, pv)
        sentence += (
            f" The one feature pointing the other way is your {label} "
            f"(~{round(pv * 100)}/100 vs the {other.biologic} group's ~{round(oth * 100)})."
        )

    margin_text = f"a {margin:g}-point edge" if margin > 0 else "a razor-thin edge"
    sentence += (
        f" On balance that gives {best.biologic} {margin_text} "
        f"({best.likelihood_pct:g}% vs {other.likelihood_pct:g}%)."
    )
    if comorbidity_label:
        sentence += (
            f" Your reported {comorbidity_label} further supports {best.biologic}, which "
            f"is also approved to treat it."
        )
    return sentence


def _explanation(
    patient: PatientFeatures,
    likelihoods: List[BiologicLikelihood],
    total_cases: int,
    exact: Optional[ExactMatch],
    scored: List[_ScoredCase],
    lifestyle_considerations: Optional[List[str]] = None,
    comorbidity_label: Optional[str] = None,
) -> Explanation:
    ranked = sorted(likelihoods, key=lambda item: item.likelihood_pct, reverse=True)
    best = ranked[0]
    top_biomarkers = _top_biomarkers(patient)

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
            rationale = _recommendation_rationale(
                patient, best, ranked[1], scored, comorbidity_label
            )

    return Explanation(
        summary=summary,
        recommendation_rationale=rationale,
        top_contributing_biomarkers=top_biomarkers,
        lifestyle_considerations=lifestyle_considerations or [],
    )


_IGA_LABELS: Dict[int, str] = {
    0: "Clear",
    1: "Almost clear",
    2: "Mild",
    3: "Moderate",
    4: "Severe",
}


def _easi_area_score(affected_body_area_pct: float) -> int:
    """Map affected body-area percentage to the EASI 0–6 area score bands."""
    pct = max(0.0, min(100.0, affected_body_area_pct))
    if pct <= 0:
        return 0
    if pct < 10:
        return 1
    if pct < 30:
        return 2
    if pct < 50:
        return 3
    if pct < 70:
        return 4
    if pct < 90:
        return 5
    return 6


def _severity_scores(patient: PatientFeatures) -> SeverityScores:
    """Estimate baseline IGA and EASI from the photo's visual biomarkers.

    Prototype proxies only: each EASI sign is scaled 0–3 from a biomarker
    (erythema→redness, induration/papulation→inflammation, excoriation→dryness/
    scaling, lichenification→skin texture), multiplied by the EASI area score
    (0–6) for a 0–72 index. IGA (0–4) is banded from the resulting EASI so the two
    stay consistent. These approximate clinician scoring; they do not replace it.
    """
    erythema = max(0.0, min(1.0, patient.erythema_score))
    induration = max(0.0, min(1.0, patient.inflammation_score))
    excoriation = max(0.0, min(1.0, patient.dryness_scaling_score))
    lichenification = max(0.0, min(1.0, patient.texture_score))

    sign_sum = (erythema + induration + excoriation + lichenification) * 3.0  # 0–12
    area_score = _easi_area_score(patient.affected_body_area_pct)  # 0–6
    easi = round(area_score * sign_sum, 1)  # 0–72

    if easi <= 0:
        iga = 0
    elif easi <= 1.0:
        iga = 1
    elif easi <= 7.0:
        iga = 2
    elif easi <= 21.0:
        iga = 3
    else:
        iga = 4

    return SeverityScores(
        iga=iga,
        iga_label=_IGA_LABELS[iga],
        easi=easi,
        severity_label=_IGA_LABELS[iga],
    )


def build_predict_response(
    image_bytes: bytes,
    age: int,
    repository: ImageReferenceRepository,
    daily_routine: str = "",
    atopic_comorbidities: str = "",
) -> PredictResponse:
    cases = repository.list_cases()
    patient_features, quality = extract_biomarkers(image_bytes)
    scored = _score_cases(patient_features, age, cases)

    likelihoods = [
        _likelihood_for_biologic(biologic, scored) for biologic in BIOLOGICS
    ]
    matched_patients = _matched_patients(scored)

    lifestyle_nudges, lifestyle_considerations = _analyze_lifestyle(daily_routine)
    comorbidity_nudges, comorbidity_considerations = _analyze_comorbidities(
        atopic_comorbidities
    )
    considerations = comorbidity_considerations + lifestyle_considerations
    nudges = _merge_nudges(lifestyle_nudges, comorbidity_nudges)
    comorbidity_label = _COMORBIDITY_LABELS.get(
        (atopic_comorbidities or "").strip().lower()
    )

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
        # Lifestyle & comorbidity only nudge the recommendation when there is no exact
        # image match (an exact match is a far stronger, image-grounded signal).
        likelihoods = _apply_lifestyle_nudges(likelihoods, nudges)

    warnings: List[str] = list(quality.warnings)
    if len(cases) < 10:
        warnings.append("small_reference_dataset")

    return PredictResponse(
        request_id=f"dm-{uuid.uuid4().hex[:8]}",
        mock=False,
        disclaimer=DISCLAIMER,
        privacy_notice=PRIVACY_NOTICE,
        patient_features=patient_features,
        severity=_severity_scores(patient_features),
        likelihoods=likelihoods,
        explanation=_explanation(
            patient_features,
            likelihoods,
            len(cases),
            exact,
            scored,
            lifestyle_considerations=considerations,
            comorbidity_label=comorbidity_label,
        ),
        heatmap=Heatmap(
            overlay_url=None,
            legend="Visual biomarker overlay is not yet rendered in this prototype.",
        ),
        matched_patients=matched_patients,
        warnings=warnings,
        exact_match=exact,
    )
