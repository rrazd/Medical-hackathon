from pathlib import Path

import pytest

from app.schemas.predict import PatientFeatures
from app.schemas.reference_case import BIOMARKER_FIELDS
from app.services.matching import (
    FEATURE_WEIGHTS,
    PatientDemographics,
    compute_feature_distances,
    weighted_gower_similarity,
)
from app.services.reference_cases import ReferenceCaseRepository

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "data"


def _features(**overrides):
    values = {
        "erythema_score": 0.0,
        "lesion_coverage_pct": 0.0,
        "texture_score": 0.0,
        "dryness_scaling_score": 0.0,
        "inflammation_score": 0.0,
        "affected_body_area_pct": 0.0,
    }
    values.update(overrides)
    return PatientFeatures(**values)


def _demographics(**overrides):
    values = {
        "age": 1,
        "sex": "female",
        "race_ethnicity": "white",
        "fitzpatrick_skin_type": "I",
        "body_area": "arm",
        "baseline_severity": "mild",
    }
    values.update(overrides)
    return PatientDemographics(**values)


def _row(case_id="DM-006"):
    return {case.row.case_id: case.row for case in ReferenceCaseRepository(DATA_ROOT).list_cases()}[case_id]


def test_feature_weights_match_research_totals_and_fields():
    assert tuple(FEATURE_WEIGHTS) == tuple(BIOMARKER_FIELDS) + (
        "age",
        "sex",
        "race_ethnicity",
        "fitzpatrick_skin_type",
        "body_area",
        "baseline_severity",
    )
    assert round(sum(FEATURE_WEIGHTS.values()), 6) == 1.0
    assert round(sum(FEATURE_WEIGHTS[field] for field in BIOMARKER_FIELDS), 6) == 0.70
    demographic_total = sum(weight for field, weight in FEATURE_WEIGHTS.items() if field not in BIOMARKER_FIELDS)
    assert round(demographic_total, 6) == 0.30


def test_fixed_range_scaling_for_numeric_and_ordinal_fields():
    row = _row("DM-006").model_copy(
        update={
            "erythema_score": 1.0,
            "lesion_coverage_pct": 100.0,
            "texture_score": 1.0,
            "dryness_scaling_score": 1.0,
            "inflammation_score": 1.0,
            "affected_body_area_pct": 100.0,
            "age": 129,
            "fitzpatrick_skin_type": "VI",
            "baseline_severity": "severe",
        }
    )

    distances = compute_feature_distances(_features(), _demographics(), row)

    assert distances["erythema_score"] == 1.0
    assert distances["lesion_coverage_pct"] == 1.0
    assert distances["texture_score"] == 1.0
    assert distances["dryness_scaling_score"] == 1.0
    assert distances["inflammation_score"] == 1.0
    assert distances["affected_body_area_pct"] == 1.0
    assert distances["age"] == 1.0
    assert distances["fitzpatrick_skin_type"] == 1.0
    assert distances["baseline_severity"] == 1.0


def test_nominal_distances_use_exact_prefer_not_and_mismatch_rules():
    row = _row("DM-006").model_copy(
        update={"sex": "female", "race_ethnicity": "prefer_not_to_say", "body_area": "arm"}
    )

    exact = compute_feature_distances(
        _features(),
        _demographics(sex="female", race_ethnicity="prefer_not_to_say", body_area="arm"),
        row,
    )
    prefer = compute_feature_distances(
        _features(),
        _demographics(sex="prefer_not_to_say", race_ethnicity="white", body_area="arm"),
        row,
    )
    mismatch = compute_feature_distances(
        _features(),
        _demographics(sex="male", race_ethnicity="white", body_area="leg"),
        row,
    )

    assert exact["sex"] == 0.0
    assert exact["race_ethnicity"] == 0.0
    assert exact["body_area"] == 0.0
    assert prefer["sex"] == 0.5
    assert prefer["race_ethnicity"] == 0.5
    assert mismatch["sex"] == 1.0
    assert mismatch["body_area"] == 1.0


def test_real_reference_rows_produce_bounded_distances_and_similarity():
    repository = ReferenceCaseRepository(DATA_ROOT)
    patient_features = _features(
        erythema_score=1.0,
        lesion_coverage_pct=100.0,
        texture_score=1.0,
        dryness_scaling_score=1.0,
        inflammation_score=1.0,
        affected_body_area_pct=100.0,
    )
    demographics = _demographics(
        age=129,
        sex="prefer_not_to_say",
        race_ethnicity="prefer_not_to_say",
        fitzpatrick_skin_type="VI",
        body_area="scalp",
        baseline_severity="severe",
    )

    for case in repository.list_cases():
        distances = compute_feature_distances(patient_features, demographics, case.row)
        assert set(distances) == set(FEATURE_WEIGHTS)
        assert all(0.0 <= value <= 1.0 for value in distances.values())
        assert 0.0 <= weighted_gower_similarity(distances) <= 1.0


def test_repository_feature_vectors_by_case_id_are_consumed_in_biomarker_order():
    repository = ReferenceCaseRepository(DATA_ROOT)
    vectors = repository.feature_vectors_by_case_id()
    case = {case.row.case_id: case for case in repository.list_cases()}["DM-006"]

    assert vectors["DM-006"] == [0.58, 20.5, 0.54, 0.59, 0.62, 10.0]
    reference_vector = dict(zip(BIOMARKER_FIELDS, vectors["DM-006"]))
    distances = compute_feature_distances(
        PatientFeatures(**reference_vector),
        _demographics(
            age=case.row.age,
            sex=case.row.sex,
            race_ethnicity=case.row.race_ethnicity,
            fitzpatrick_skin_type=case.row.fitzpatrick_skin_type,
            body_area=case.row.body_area,
            baseline_severity=case.row.baseline_severity,
        ),
        case.row,
        reference_vector,
    )

    assert all(distances[field] == 0.0 for field in BIOMARKER_FIELDS)
