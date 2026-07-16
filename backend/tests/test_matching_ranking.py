from pathlib import Path

from app.schemas.predict import MatchedPatient, PatientFeatures
from app.schemas.reference_case import ReferenceCase, ReferenceCaseRow
from app.services.matching import PatientDemographics, run_matching
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


def test_real_dm006_patient_ranks_dm006_first_with_exact_similarity():
    repository = ReferenceCaseRepository(DATA_ROOT)
    rows = {case.row.case_id: case.row for case in repository.list_cases()}
    dm006 = rows["DM-006"]

    result = run_matching(_features_from_row(dm006), _demographics_from_row(dm006), repository=repository, top_k=5)

    assert len(result.matched_patients) == 5
    assert result.matched_patients[0].case_id == "DM-006"
    assert result.matched_patients[0].similarity == 1.0
    assert set(result.contributions_by_case_id) == {match.case_id for match in result.matched_patients}
    assert all(result.contributions_by_case_id["DM-006"][index].distance == 0.0 for index in range(12))


def test_top_k_is_honored_against_real_ten_row_dataset():
    repository = ReferenceCaseRepository(DATA_ROOT)
    dm006 = {case.row.case_id: case.row for case in repository.list_cases()}["DM-006"]

    assert len(run_matching(_features_from_row(dm006), _demographics_from_row(dm006), repository, top_k=3).matched_patients) == 3
    assert len(run_matching(_features_from_row(dm006), _demographics_from_row(dm006), repository, top_k=20).matched_patients) == 10


def test_tie_breaking_uses_similarity_biomarker_similarity_then_case_id():
    patient_row = _row("DM-100")
    later = _row("DM-102")
    earlier = _row("DM-101")
    repository = InMemoryRepository([later, earlier])

    result = run_matching(_features_from_row(patient_row), _demographics_from_row(patient_row), repository=repository, top_k=2)

    assert [match.case_id for match in result.matched_patients] == ["DM-101", "DM-102"]


def test_biomarker_dominance_ranks_biomarker_close_above_demographic_same():
    patient_features = PatientFeatures(
        erythema_score=0.0,
        lesion_coverage_pct=0.0,
        texture_score=0.0,
        dryness_scaling_score=0.0,
        inflammation_score=0.0,
        affected_body_area_pct=0.0,
    )
    demographics = PatientDemographics(
        age=1,
        sex="female",
        race_ethnicity="white",
        fitzpatrick_skin_type="I",
        body_area="arm",
        baseline_severity="mild",
    )
    biomarker_close = _row(
        "DM-201",
        age=129,
        sex="male",
        race_ethnicity="asian",
        fitzpatrick_skin_type="VI",
        body_area="scalp",
        baseline_severity="severe",
        erythema_score=0.0,
        lesion_coverage_pct=0.0,
        texture_score=0.0,
        dryness_scaling_score=0.0,
        inflammation_score=0.0,
        affected_body_area_pct=0.0,
    )
    demographic_same = _row(
        "DM-202",
        age=1,
        sex="female",
        race_ethnicity="white",
        fitzpatrick_skin_type="I",
        body_area="arm",
        baseline_severity="mild",
        erythema_score=1.0,
        lesion_coverage_pct=100.0,
        texture_score=1.0,
        dryness_scaling_score=1.0,
        inflammation_score=1.0,
        affected_body_area_pct=100.0,
    )

    result = run_matching(patient_features, demographics, repository=InMemoryRepository([demographic_same, biomarker_close]), top_k=2)

    assert [match.case_id for match in result.matched_patients] == ["DM-201", "DM-202"]
    assert result.matched_patients[0].similarity > result.matched_patients[1].similarity


def test_matched_patient_output_shape_safe_images_and_reasons():
    repository = ReferenceCaseRepository(DATA_ROOT)
    dm006 = {case.row.case_id: case.row for case in repository.list_cases()}["DM-006"]

    result = run_matching(_features_from_row(dm006), _demographics_from_row(dm006), repository=repository, top_k=5)

    expected_fields = set(MatchedPatient.model_fields)
    for match in result.matched_patients:
        assert set(match.model_dump()) == expected_fields
        assert 0.0 <= match.similarity <= 1.0
        assert match.before_image_url is None or not Path(match.before_image_url).is_absolute()
        assert match.after_image_url is None or not Path(match.after_image_url).is_absolute()
        assert 2 <= len(match.matching_reasons) <= 4
        assert all(reason for reason in match.matching_reasons)
