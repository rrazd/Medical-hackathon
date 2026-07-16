from pathlib import Path

import pytest

from app.schemas.predict import PatientFeatures
from app.schemas.reference_case import ReferenceCase, ReferenceCaseRow
from app.services.matching import (
    LikelihoodEvidence,
    PatientDemographics,
    aggregate_biologic_likelihood,
    run_matching,
)
from app.services.reference_cases import ReferenceCaseRepository

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "data"


class InMemoryRepository:
    def __init__(self, rows):
        self._cases = [
            ReferenceCase(
                row=row,
                before_image_file=Path(f"images/{row.case_id}/before.jpg"),
                after_image_file=Path(f"images/{row.case_id}/after.jpg"),
            )
            for row in rows
        ]

    def list_cases(self):
        return list(self._cases)

    def feature_vectors_by_case_id(self):
        return {case.row.case_id: case.row.biomarker_vector() for case in self._cases}


def _row(case_id, **overrides):
    values = {
        "case_id": case_id,
        "biologic": "Dupixent",
        "outcome_label": "responder",
        "outcome_score": 1.0,
        "age": 40,
        "sex": "female",
        "race_ethnicity": "white",
        "fitzpatrick_skin_type": "III",
        "body_area": "arm",
        "baseline_severity": "moderate",
        "before_image_path": f"images/{case_id}/before.jpg",
        "after_image_path": f"images/{case_id}/after.jpg",
        "erythema_score": 0.5,
        "lesion_coverage_pct": 20.0,
        "texture_score": 0.5,
        "dryness_scaling_score": 0.5,
        "inflammation_score": 0.5,
        "affected_body_area_pct": 10.0,
        "followup_weeks": 16,
    }
    values.update(overrides)
    return ReferenceCaseRow.model_validate(values)


def _expected_range(similarities, outcomes):
    weights = [similarity**2 for similarity in similarities]
    weight_sum = sum(weights)
    mean = sum(weight * outcome for weight, outcome in zip(weights, outcomes)) / weight_sum
    variance = sum(weight * ((outcome - mean) ** 2) for weight, outcome in zip(weights, outcomes)) / weight_sum
    sparse_penalty = max(0, 3 - len(similarities)) * 0.10
    weak_similarity_penalty = 0.08 if (sum(similarities) / len(similarities)) < 0.70 else 0.0
    half_width = min(0.50, max(0.15, variance**0.5) + sparse_penalty + weak_similarity_penalty)
    return (
        mean,
        round(100 * max(0.0, mean - half_width)),
        round(100 * min(1.0, mean + half_width)),
    )


def _features_from_row(row):
    return PatientFeatures(
        erythema_score=row.erythema_score,
        lesion_coverage_pct=row.lesion_coverage_pct,
        texture_score=row.texture_score,
        dryness_scaling_score=row.dryness_scaling_score,
        inflammation_score=row.inflammation_score,
        affected_body_area_pct=row.affected_body_area_pct,
    )


def _demographics_from_row(row):
    return PatientDemographics(
        age=row.age,
        sex=row.sex,
        race_ethnicity=row.race_ethnicity,
        fitzpatrick_skin_type=row.fitzpatrick_skin_type,
        body_area=row.body_area,
        baseline_severity=row.baseline_severity,
    )


def test_aggregate_biologic_likelihood_uses_similarity_squared_weighted_mean_and_range_formula():
    similarities = [0.90, 0.75, 0.65]
    outcomes = [1.0, 0.5, 0.0]
    evidence = [
        LikelihoodEvidence("Dupixent", similarity, outcome)
        for similarity, outcome in zip(similarities, outcomes)
    ] + [LikelihoodEvidence("Ebglyss", 1.0, 0.0)]
    expected_mean, expected_low, expected_high = _expected_range(similarities, outcomes)

    likelihood, likelihood_range = aggregate_biologic_likelihood("Dupixent", evidence)

    assert likelihood.weighted_outcome_score == pytest.approx(round(expected_mean, 3))
    assert likelihood.likelihood_pct == round(100 * expected_mean)
    assert likelihood_range == (expected_low, expected_high)
    assert expected_low <= likelihood.likelihood_pct <= expected_high
    assert likelihood.matched_case_count == 3
    assert likelihood.confidence_label == "low"
    assert "Evidence range:" in likelihood.caveat
    assert "not a treatment recommendation" in likelihood.caveat


def test_aggregate_biologic_likelihood_widens_sparse_and_weak_similarity_ranges():
    single_likelihood, single_range = aggregate_biologic_likelihood(
        "Dupixent",
        [LikelihoodEvidence("Dupixent", 0.68, 0.5)],
    )
    three_likelihood, three_range = aggregate_biologic_likelihood(
        "Dupixent",
        [
            LikelihoodEvidence("Dupixent", 0.91, 0.5),
            LikelihoodEvidence("Dupixent", 0.88, 0.5),
            LikelihoodEvidence("Dupixent", 0.86, 0.5),
        ],
    )

    assert single_likelihood.likelihood_pct == three_likelihood.likelihood_pct == 50
    assert (single_range[1] - single_range[0]) > (three_range[1] - three_range[0])


def test_aggregate_biologic_likelihood_zero_usable_same_biologic_matches_is_neutral_placeholder():
    likelihood, likelihood_range = aggregate_biologic_likelihood(
        "Ebglyss",
        [
            LikelihoodEvidence("Dupixent", 0.99, 1.0),
            LikelihoodEvidence("Ebglyss", 0.59, 0.0),
        ],
    )

    assert likelihood.weighted_outcome_score == 0.5
    assert likelihood.likelihood_pct == 50
    assert likelihood_range == (0, 100)
    assert likelihood.matched_case_count == 0
    assert likelihood.confidence_label == "insufficient"
    assert "neutral placeholder" in likelihood.caveat
    assert "not an estimate" in likelihood.caveat


def test_run_matching_real_dataset_returns_range_backed_likelihoods_for_both_biologics():
    repository = ReferenceCaseRepository(DATA_ROOT)
    rows = {case.row.case_id: case.row for case in repository.list_cases()}
    dm006 = rows["DM-006"]

    result = run_matching(_features_from_row(dm006), _demographics_from_row(dm006), repository=repository, top_k=5)

    assert [likelihood.biologic for likelihood in result.likelihoods] == ["Dupixent", "Ebglyss"]
    assert set(result.likelihood_ranges_pct) == {"Dupixent", "Ebglyss"}
    for likelihood in result.likelihoods:
        low, high = result.likelihood_ranges_pct[likelihood.biologic]
        assert 0 <= low <= likelihood.likelihood_pct <= high <= 100
        assert f"Evidence range: {low}-{high}%." in likelihood.caveat
        assert "similar" in likelihood.caveat
        assert "prototype estimate" in likelihood.caveat
        assert "not a treatment recommendation" in likelihood.caveat


def test_run_matching_counts_only_same_biologic_top_five_matches_at_similarity_floor():
    repository = ReferenceCaseRepository(DATA_ROOT)
    rows = {case.row.case_id: case.row for case in repository.list_cases()}
    dm006 = rows["DM-006"]
    result = run_matching(_features_from_row(dm006), _demographics_from_row(dm006), repository=repository, top_k=5)

    expected_counts = {"Dupixent": 0, "Ebglyss": 0}
    for match in result.matched_patients:
        if match.similarity >= 0.60:
            expected_counts[match.biologic_used] += 1

    assert {likelihood.biologic: likelihood.matched_case_count for likelihood in result.likelihoods} == expected_counts


def test_run_matching_likelihood_weighted_values_match_real_top_five_outcomes():
    repository = ReferenceCaseRepository(DATA_ROOT)
    rows = {case.row.case_id: case.row for case in repository.list_cases()}
    dm006 = rows["DM-006"]
    result = run_matching(_features_from_row(dm006), _demographics_from_row(dm006), repository=repository, top_k=5)
    likelihoods = {likelihood.biologic: likelihood for likelihood in result.likelihoods}

    for biologic, likelihood in likelihoods.items():
        usable = [
            match
            for match in result.matched_patients
            if match.biologic_used == biologic and match.similarity >= 0.60
        ]
        if not usable:
            assert likelihood.weighted_outcome_score == 0.5
            continue
        weights = [match.similarity**2 for match in usable]
        expected = sum(weight * match.outcome_score for weight, match in zip(weights, usable)) / sum(weights)
        assert likelihood.weighted_outcome_score == pytest.approx(round(expected, 3))


def test_run_matching_zero_usable_biologic_support_does_not_crash():
    patient = _row("DM-100")
    repository = InMemoryRepository([patient])

    result = run_matching(_features_from_row(patient), _demographics_from_row(patient), repository=repository, top_k=1)
    likelihoods = {likelihood.biologic: likelihood for likelihood in result.likelihoods}

    assert likelihoods["Dupixent"].matched_case_count == 1
    assert likelihoods["Ebglyss"].matched_case_count == 0
    assert likelihoods["Ebglyss"].weighted_outcome_score == 0.5
    assert likelihoods["Ebglyss"].likelihood_pct == 50
    assert likelihoods["Ebglyss"].confidence_label == "insufficient"
    assert result.likelihood_ranges_pct["Ebglyss"] == (0, 100)
