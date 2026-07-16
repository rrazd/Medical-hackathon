import csv
from pathlib import Path

import pytest

from app.schemas.predict import PatientFeatures
from app.services.reference_cases import (
    InvalidReferenceCaseError,
    MissingImageFileError,
    ReferenceCaseRepository,
    ReferenceDatasetSchemaError,
    UnsafeImagePathError,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "data"
CSV_PATH = DATA_ROOT / "reference_cases.csv"

CANONICAL_HEADER = [
    "case_id",
    "biologic",
    "outcome_label",
    "outcome_score",
    "age",
    "sex",
    "race_ethnicity",
    "fitzpatrick_skin_type",
    "body_area",
    "baseline_severity",
    "before_image_path",
    "after_image_path",
    "erythema_score",
    "lesion_coverage_pct",
    "texture_score",
    "dryness_scaling_score",
    "inflammation_score",
    "affected_body_area_pct",
    "followup_weeks",
]
BIOMARKER_FIELDS = list(PatientFeatures.model_fields)


def _seed_row(case_id="DM-999", **overrides):
    row = {
        "case_id": case_id,
        "biologic": "Dupixent",
        "outcome_label": "responder",
        "outcome_score": "1.0",
        "age": "42",
        "sex": "female",
        "race_ethnicity": "white",
        "fitzpatrick_skin_type": "III",
        "body_area": "arm",
        "baseline_severity": "moderate",
        "before_image_path": f"images/{case_id}/before.jpg",
        "after_image_path": f"images/{case_id}/after.jpg",
        "erythema_score": "0.72",
        "lesion_coverage_pct": "18.5",
        "texture_score": "0.58",
        "dryness_scaling_score": "0.61",
        "inflammation_score": "0.70",
        "affected_body_area_pct": "8.5",
        "followup_weeks": "16",
    }
    row.update(overrides)
    return row


def _write_dataset(tmp_path, rows, header=CANONICAL_HEADER, create_images=True):
    data_root = tmp_path / "data"
    data_root.mkdir()
    with (data_root / "reference_cases.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    if create_images:
        for row in rows:
            for image_field in ("before_image_path", "after_image_path"):
                image_path = row.get(image_field)
                if image_path and not Path(image_path).is_absolute() and ".." not in Path(image_path).parts:
                    file_path = data_root / image_path
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_bytes(b"synthetic image bytes")
    return data_root


def test_repository_loads_real_seed_csv_as_typed_records():
    repository = ReferenceCaseRepository(DATA_ROOT)

    cases = repository.list_cases()

    assert len(cases) == 10
    assert cases[0].row.case_id == "DM-001"
    assert cases[0].before_image_file == (DATA_ROOT / "images/DM-001/before.jpg").resolve()
    assert cases[0].after_image_file == (DATA_ROOT / "images/DM-001/after.jpg").resolve()
    assert [case.row.case_id for case in cases] == [f"DM-{index:03d}" for index in range(1, 11)]


@pytest.mark.parametrize(
    ("header", "expected_message"),
    [
        ([column for column in CANONICAL_HEADER if column != "followup_weeks"], "followup_weeks"),
        (CANONICAL_HEADER + ["unexpected_column"], "unexpected_column"),
    ],
)
def test_repository_rejects_missing_or_unexpected_csv_columns(tmp_path, header, expected_message):
    data_root = _write_dataset(tmp_path, [_seed_row()], header=header)

    with pytest.raises(ReferenceDatasetSchemaError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert expected_message in str(error.value)


def test_repository_rejects_bad_enum_values_with_row_context(tmp_path):
    row = _seed_row(case_id="DM-998", biologic="UnknownDrug")
    data_root = _write_dataset(tmp_path, [row])

    with pytest.raises(InvalidReferenceCaseError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert "line 2" in str(error.value)
    assert "DM-998" in str(error.value)
    assert "biologic" in str(error.value)


def test_repository_rejects_duplicate_case_ids(tmp_path):
    data_root = _write_dataset(tmp_path, [_seed_row(case_id="DM-994"), _seed_row(case_id="DM-994")])

    with pytest.raises(InvalidReferenceCaseError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert "Duplicate" in str(error.value)
    assert "DM-994" in str(error.value)


def test_repository_rejects_missing_image_files_with_case_context(tmp_path):
    row = _seed_row(case_id="DM-997")
    data_root = _write_dataset(tmp_path, [row], create_images=False)

    with pytest.raises(MissingImageFileError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert "DM-997" in str(error.value)
    assert "before_image_path" in str(error.value)


@pytest.mark.parametrize(
    ("field", "bad_path", "message"),
    [
        ("before_image_path", "/absolute/before.jpg", "absolute"),
        ("before_image_path", "images/DM-996/../../secret.jpg", "traversal"),
        ("after_image_path", "images/DM-OTHER/after.jpg", "images/DM-996"),
    ],
)
def test_repository_rejects_absolute_traversal_and_wrong_case_image_paths(tmp_path, field, bad_path, message):
    row = _seed_row(case_id="DM-996", **{field: bad_path})
    data_root = _write_dataset(tmp_path, [row])

    with pytest.raises(UnsafeImagePathError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert "DM-996" in str(error.value)
    assert message in str(error.value)


def test_repository_rejects_identical_before_and_after_paths(tmp_path):
    row = _seed_row(case_id="DM-995", after_image_path="images/DM-995/before.jpg")
    data_root = _write_dataset(tmp_path, [row])

    with pytest.raises(UnsafeImagePathError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert "same file" in str(error.value)
    assert "DM-995" in str(error.value)


def test_repository_feature_vectors_have_expected_shape_and_cache():
    repository = ReferenceCaseRepository(DATA_ROOT)

    vectors = repository.feature_vectors_by_case_id()
    cases_after_vectors = repository.list_cases()

    assert list(vectors) == [f"DM-{index:03d}" for index in range(1, 11)]
    assert len(vectors) == len(cases_after_vectors) == 10
    assert all(len(vector) == len(BIOMARKER_FIELDS) for vector in vectors.values())
    assert vectors["DM-001"] == [0.72, 18.5, 0.58, 0.61, 0.70, 8.5]


def test_repository_caches_successful_loads_without_rereading_csv(tmp_path):
    data_root = _write_dataset(tmp_path, [_seed_row(case_id="DM-993")])
    repository = ReferenceCaseRepository(data_root)

    first_cases = repository.list_cases()
    first_vectors = repository.feature_vectors_by_case_id()
    (data_root / "reference_cases.csv").write_text("bad_column\nbad_value\n")

    assert repository.list_cases()[0].row.case_id == first_cases[0].row.case_id
    assert repository.feature_vectors_by_case_id() == first_vectors
