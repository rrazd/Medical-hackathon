from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from app.schemas.predict import BiologicLikelihood, MatchedPatient, PatientFeatures
from app.schemas.reference_case import BIOMARKER_FIELDS, ReferenceCase, ReferenceCaseRow
from app.services.reference_cases import ReferenceCaseRepository

BIOMARKER_FIELD_SET = set(BIOMARKER_FIELDS)
DEMOGRAPHIC_CONTEXT_FIELDS = (
    "age",
    "sex",
    "race_ethnicity",
    "fitzpatrick_skin_type",
    "body_area",
    "baseline_severity",
)
FEATURE_WEIGHTS: Dict[str, float] = {
    "erythema_score": 0.10,
    "lesion_coverage_pct": 0.14,
    "texture_score": 0.11,
    "dryness_scaling_score": 0.11,
    "inflammation_score": 0.14,
    "affected_body_area_pct": 0.10,
    "age": 0.04,
    "sex": 0.03,
    "race_ethnicity": 0.02,
    "fitzpatrick_skin_type": 0.08,
    "body_area": 0.08,
    "baseline_severity": 0.05,
}

FITZPATRICK_ORDINAL = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6}
SEVERITY_ORDINAL = {"mild": 0.0, "moderate": 0.5, "severe": 1.0}
PERCENT_FIELDS = {"lesion_coverage_pct", "affected_body_area_pct"}
BIOLOGICS = ("Dupixent", "Ebglyss")
LIKELIHOOD_SIMILARITY_FLOOR = 0.60
REASON_FIELD_PRIORITY = {
    "inflammation_score": 0,
    "lesion_coverage_pct": 1,
    "erythema_score": 2,
    "texture_score": 3,
    "dryness_scaling_score": 4,
    "affected_body_area_pct": 5,
    "body_area": 6,
    "fitzpatrick_skin_type": 7,
    "baseline_severity": 8,
    "age": 9,
    "sex": 10,
    "race_ethnicity": 99,
}


@dataclass(frozen=True)
class PatientDemographics:
    age: int
    sex: str
    race_ethnicity: str
    fitzpatrick_skin_type: str
    body_area: str
    baseline_severity: str


@dataclass(frozen=True)
class FeatureContribution:
    field: str
    patient_value: object
    reference_value: object
    distance: float
    weight: float
    contribution: float
    group: str


@dataclass(frozen=True)
class MatchingResult:
    matched_patients: List[MatchedPatient]
    likelihoods: List[BiologicLikelihood]
    warnings: List[str]
    contributions_by_case_id: Dict[str, List[FeatureContribution]]
    likelihood_ranges_pct: Dict[str, Tuple[int, int]]


@dataclass(frozen=True)
class _ScoredCase:
    case: ReferenceCase
    similarity: float
    biomarker_similarity: float
    contributions: List[FeatureContribution]


@dataclass(frozen=True)
class LikelihoodEvidence:
    biologic: str
    similarity: float
    outcome_score: float


def compute_feature_distances(
    patient_features: PatientFeatures,
    demographics: PatientDemographics,
    reference_row: ReferenceCaseRow,
    reference_biomarker_values: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    patient_values = _patient_values(patient_features, demographics)
    reference_values = _reference_values(reference_row, reference_biomarker_values)
    return {
        field: _feature_distance(field, patient_values[field], reference_values[field])
        for field in FEATURE_WEIGHTS
    }


def weighted_gower_similarity(
    distances: Mapping[str, float],
    weights: Mapping[str, float] = FEATURE_WEIGHTS,
) -> float:
    weighted_distance = float(
        np.sum(np.array([weights[field] * distances[field] for field in weights], dtype=float))
    )
    return round(max(0.0, min(1.0, 1.0 - weighted_distance)), 3)


def aggregate_biologic_likelihood(
    biologic: str,
    evidence: Sequence[LikelihoodEvidence],
) -> Tuple[BiologicLikelihood, Tuple[int, int]]:
    usable = [
        item
        for item in evidence
        if item.biologic == biologic and item.similarity >= LIKELIHOOD_SIMILARITY_FLOOR
    ]
    if not usable:
        likelihood = BiologicLikelihood(
            biologic=biologic,
            likelihood_pct=50,
            confidence_label="insufficient",
            matched_case_count=0,
            weighted_outcome_score=0.5,
            caveat=(
                "Evidence range: 0-100%. Insufficient evidence: no similar "
                f"{biologic} reference cases met the 0.60 similarity floor; "
                "50% is a neutral placeholder, not an estimate. This is a prototype estimate, "
                "not a treatment recommendation."
            ),
        )
        return likelihood, (0, 100)

    similarities = np.array([item.similarity for item in usable], dtype=float)
    outcomes = np.array([item.outcome_score for item in usable], dtype=float)
    weights = similarities**2
    weighted_outcome_score = float(np.sum(weights * outcomes) / np.sum(weights))
    weighted_std = float(np.sqrt(np.sum(weights * (outcomes - weighted_outcome_score) ** 2) / np.sum(weights)))
    average_similarity = float(np.mean(similarities))
    sparse_penalty = max(0, 3 - len(usable)) * 0.10
    weak_similarity_penalty = 0.08 if average_similarity < 0.70 else 0.0
    half_width = min(0.50, max(0.15, weighted_std) + sparse_penalty + weak_similarity_penalty)
    low_pct = round(100 * max(0.0, weighted_outcome_score - half_width))
    high_pct = round(100 * min(1.0, weighted_outcome_score + half_width))
    likelihood_pct = round(100 * weighted_outcome_score)
    support_word = "case" if len(usable) == 1 else "cases"
    caveat = (
        f"Evidence range: {low_pct}-{high_pct}%. Based on {len(usable)} similar "
        f"{biologic} reference {support_word}; prototype estimate, not a treatment recommendation."
    )
    if len(usable) < 3 or average_similarity < 0.70:
        caveat += " Range widened for sparse or weaker same-biologic support."
    likelihood = BiologicLikelihood(
        biologic=biologic,
        likelihood_pct=likelihood_pct,
        confidence_label="low",
        matched_case_count=len(usable),
        weighted_outcome_score=round(weighted_outcome_score, 3),
        caveat=caveat,
    )
    return likelihood, (low_pct, high_pct)


def run_matching(
    patient_features: PatientFeatures,
    demographics: PatientDemographics,
    repository: Optional[ReferenceCaseRepository] = None,
    top_k: int = 5,
) -> MatchingResult:
    repo = repository or ReferenceCaseRepository()
    feature_vectors = repo.feature_vectors_by_case_id()
    cases = repo.list_cases()
    scored_cases = [_score_case(patient_features, demographics, case, feature_vectors) for case in cases]
    ranked = sorted(
        scored_cases,
        key=lambda scored: (-scored.similarity, -scored.biomarker_similarity, scored.case.row.case_id),
    )
    selected = ranked[: max(0, min(top_k, len(ranked)))]
    matches = [_to_matched_patient(scored) for scored in selected]
    warnings = _build_match_warnings(selected)
    evidence = [
        LikelihoodEvidence(
            biologic=scored.case.row.biologic,
            similarity=scored.similarity,
            outcome_score=scored.case.row.outcome_score,
        )
        for scored in selected
    ]
    likelihood_pairs = [aggregate_biologic_likelihood(biologic, evidence) for biologic in BIOLOGICS]
    return MatchingResult(
        matched_patients=matches,
        likelihoods=[likelihood for likelihood, _likelihood_range in likelihood_pairs],
        warnings=warnings,
        contributions_by_case_id={scored.case.row.case_id: scored.contributions for scored in selected},
        likelihood_ranges_pct={
            likelihood.biologic: likelihood_range
            for likelihood, likelihood_range in likelihood_pairs
        },
    )


def _score_case(
    patient_features: PatientFeatures,
    demographics: PatientDemographics,
    case: ReferenceCase,
    feature_vectors: Mapping[str, Sequence[float]],
) -> _ScoredCase:
    reference_biomarkers = _biomarkers_from_vector(case.row.case_id, case.row, feature_vectors)
    distances = compute_feature_distances(patient_features, demographics, case.row, reference_biomarkers)
    similarity = weighted_gower_similarity(distances)
    biomarker_similarity = _biomarker_similarity(distances)
    contributions = _build_contributions(
        patient_features,
        demographics,
        case.row,
        reference_biomarkers,
        distances,
    )
    return _ScoredCase(case=case, similarity=similarity, biomarker_similarity=biomarker_similarity, contributions=contributions)


def _biomarkers_from_vector(
    case_id: str,
    row: ReferenceCaseRow,
    feature_vectors: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
    vector = feature_vectors.get(case_id)
    if vector is None or len(vector) != len(BIOMARKER_FIELDS):
        return {field: float(getattr(row, field)) for field in BIOMARKER_FIELDS}
    return {field: float(value) for field, value in zip(BIOMARKER_FIELDS, vector)}


def _patient_values(patient_features: PatientFeatures, demographics: PatientDemographics) -> Dict[str, object]:
    values = {field: float(getattr(patient_features, field)) for field in BIOMARKER_FIELDS}
    values.update(
        {
            "age": demographics.age,
            "sex": demographics.sex,
            "race_ethnicity": demographics.race_ethnicity,
            "fitzpatrick_skin_type": demographics.fitzpatrick_skin_type,
            "body_area": demographics.body_area,
            "baseline_severity": demographics.baseline_severity,
        }
    )
    return values


def _reference_values(
    reference_row: ReferenceCaseRow,
    reference_biomarker_values: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    biomarker_values = reference_biomarker_values or {
        field: float(getattr(reference_row, field)) for field in BIOMARKER_FIELDS
    }
    values = {field: float(biomarker_values[field]) for field in BIOMARKER_FIELDS}
    values.update({field: getattr(reference_row, field) for field in DEMOGRAPHIC_CONTEXT_FIELDS})
    return values


def _feature_distance(field: str, patient_value: object, reference_value: object) -> float:
    if field in PERCENT_FIELDS:
        return _clip_unit(abs(float(patient_value) - float(reference_value)) / 100.0)
    if field in BIOMARKER_FIELD_SET:
        return _clip_unit(abs(float(patient_value) - float(reference_value)))
    if field == "age":
        return _clip_unit(abs(float(patient_value) - float(reference_value)) / 128.0)
    if field == "fitzpatrick_skin_type":
        return _clip_unit(abs(_fitzpatrick_value(patient_value) - _fitzpatrick_value(reference_value)) / 5.0)
    if field == "baseline_severity":
        return _clip_unit(abs(_severity_value(patient_value) - _severity_value(reference_value)))
    if field in {"sex", "race_ethnicity"}:
        return _prefer_not_nominal_distance(patient_value, reference_value)
    if field == "body_area":
        return 0.0 if _normalize_token(patient_value) == _normalize_token(reference_value) else 1.0
    raise KeyError(f"Unsupported matching field: {field}")


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_token(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _fitzpatrick_value(value: object) -> int:
    token = str(value).strip().upper()
    if token not in FITZPATRICK_ORDINAL:
        raise ValueError(f"Unsupported Fitzpatrick skin type: {value}")
    return FITZPATRICK_ORDINAL[token]


def _severity_value(value: object) -> float:
    token = _normalize_token(value)
    if token not in SEVERITY_ORDINAL:
        raise ValueError(f"Unsupported baseline severity: {value}")
    return SEVERITY_ORDINAL[token]


def _prefer_not_nominal_distance(patient_value: object, reference_value: object) -> float:
    patient = _normalize_token(patient_value)
    reference = _normalize_token(reference_value)
    if patient == reference:
        return 0.0
    if patient == "prefer_not_to_say" or reference == "prefer_not_to_say":
        return 0.5
    return 1.0


def _biomarker_similarity(distances: Mapping[str, float]) -> float:
    biomarker_weight_total = sum(FEATURE_WEIGHTS[field] for field in BIOMARKER_FIELDS)
    biomarker_distance = sum(FEATURE_WEIGHTS[field] * distances[field] for field in BIOMARKER_FIELDS) / biomarker_weight_total
    return round(max(0.0, min(1.0, 1.0 - biomarker_distance)), 6)


def _build_contributions(
    patient_features: PatientFeatures,
    demographics: PatientDemographics,
    reference_row: ReferenceCaseRow,
    reference_biomarkers: Mapping[str, float],
    distances: Mapping[str, float],
) -> List[FeatureContribution]:
    patient_values = _patient_values(patient_features, demographics)
    reference_values = _reference_values(reference_row, reference_biomarkers)
    return [
        FeatureContribution(
            field=field,
            patient_value=patient_values[field],
            reference_value=reference_values[field],
            distance=round(float(distances[field]), 6),
            weight=FEATURE_WEIGHTS[field],
            contribution=round(FEATURE_WEIGHTS[field] * (1.0 - float(distances[field])), 6),
            group="biomarker" if field in BIOMARKER_FIELD_SET else "demographic_context",
        )
        for field in FEATURE_WEIGHTS
    ]


def _to_matched_patient(scored: _ScoredCase) -> MatchedPatient:
    row = scored.case.row
    return MatchedPatient(
        case_id=row.case_id,
        similarity=scored.similarity,
        biologic_used=row.biologic,
        outcome_label=row.outcome_label,
        outcome_score=row.outcome_score,
        demographic_summary=_demographic_summary(row),
        matching_reasons=_matching_reasons(scored.contributions),
        before_image_url=None,
        after_image_url=None,
    )


def _demographic_summary(row: ReferenceCaseRow) -> str:
    return (
        f"Age {row.age}, {row.sex}, Fitzpatrick {row.fitzpatrick_skin_type}, "
        f"{row.body_area}, {row.baseline_severity} baseline severity"
    )


def _matching_reasons(contributions: Iterable[FeatureContribution]) -> List[str]:
    sorted_contributions = sorted(
        contributions,
        key=lambda contribution: (
            -contribution.contribution,
            REASON_FIELD_PRIORITY.get(contribution.field, 50),
            contribution.field,
        ),
    )
    reasons: List[str] = []
    for contribution in sorted_contributions:
        if contribution.contribution <= 0.0:
            continue
        if contribution.field == "race_ethnicity" and len(reasons) < 3:
            continue
        reason = _format_reason(contribution)
        if reason and reason not in reasons:
            reasons.append(reason)
        if len(reasons) >= 4:
            break
    if len(reasons) < 2:
        for contribution in sorted_contributions:
            reason = _format_reason(contribution)
            if reason and reason not in reasons:
                reasons.append(reason)
            if len(reasons) >= 2:
                break
    return reasons[:4]


def _format_reason(contribution: FeatureContribution) -> str:
    field = contribution.field
    patient_value = contribution.patient_value
    reference_value = contribution.reference_value
    if field == "inflammation_score":
        return f"similar inflammation score ({_fmt_number(patient_value)} vs {_fmt_number(reference_value)})"
    if field == "lesion_coverage_pct":
        return f"similar lesion coverage ({_fmt_pct(patient_value)} vs {_fmt_pct(reference_value)})"
    if field == "erythema_score":
        return f"similar erythema score ({_fmt_number(patient_value)} vs {_fmt_number(reference_value)})"
    if field == "texture_score":
        return f"similar texture score ({_fmt_number(patient_value)} vs {_fmt_number(reference_value)})"
    if field == "dryness_scaling_score":
        return f"similar dryness/scaling score ({_fmt_number(patient_value)} vs {_fmt_number(reference_value)})"
    if field == "affected_body_area_pct":
        return f"similar affected body area ({_fmt_pct(patient_value)} vs {_fmt_pct(reference_value)})"
    if field == "body_area":
        return f"same body area: {reference_value}" if contribution.distance == 0.0 else f"closest available body area: {reference_value}"
    if field == "fitzpatrick_skin_type":
        return f"similar Fitzpatrick skin type ({patient_value} vs {reference_value})"
    if field == "baseline_severity":
        return f"same baseline severity: {reference_value}" if contribution.distance == 0.0 else f"similar baseline severity ({patient_value} vs {reference_value})"
    if field == "age":
        return f"similar age ({patient_value} vs {reference_value})"
    if field == "sex" and contribution.distance == 0.0:
        return f"same sex recorded: {reference_value}"
    if field == "race_ethnicity" and contribution.distance == 0.0:
        return f"same race/ethnicity recorded: {reference_value}"
    return ""


def _fmt_number(value: object) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def _fmt_pct(value: object) -> str:
    return f"{float(value):.1f}%"


def _build_match_warnings(selected: Sequence[_ScoredCase]) -> List[str]:
    if not selected:
        return ["no_reference_cases: no reference cases were available for matching."]
    warnings: List[str] = []
    if selected[0].similarity < 0.55:
        warnings.append("weak_overall_match: closest available reference cases are not strongly similar; interpret ranges cautiously.")
    if any(scored.similarity < 0.60 for scored in selected):
        warnings.append("some_matches_below_similarity_floor: displayed closest cases may not contribute to likelihood evidence.")
    return warnings


__all__ = [
    "FEATURE_WEIGHTS",
    "LIKELIHOOD_SIMILARITY_FLOOR",
    "LikelihoodEvidence",
    "FeatureContribution",
    "MatchingResult",
    "PatientDemographics",
    "aggregate_biologic_likelihood",
    "compute_feature_distances",
    "run_matching",
    "weighted_gower_similarity",
]
