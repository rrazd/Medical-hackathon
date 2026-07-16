import csv
from pathlib import Path

import pytest

from app.services.reference_cases import (
    InvalidReferenceCaseError,
    ReferenceCaseRepository,
    UnsafeImagePathError,
)


BACKEND_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = BACKEND_ROOT / "data"

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


def _seed_row(case_id="DM-990", **overrides):
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


def _write_dataset(tmp_path, rows, create_images=True):
    data_root = tmp_path / "data"
    data_root.mkdir()
    with (data_root / "reference_cases.csv").open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CANONICAL_HEADER)
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


@pytest.mark.parametrize(
    ("field", "pii_value", "description"),
    [
        ("race_ethnicity", "Jane Smith", "likely personal name"),
        ("case_id", "123-45-6789", "SSN-like value"),
        ("body_area", "555-123-4567", "phone-like value"),
        ("baseline_severity", "demo@example.com", "email address"),
        ("biologic", "MRN123456", "MRN-like value"),
        ("outcome_label", "DOB 01/15/1980", "PII label"),
        ("after_image_path", "images/DM-990/01/15/1980.jpg", "date-like value"),
    ],
)
def test_repository_rejects_pii_like_values_in_any_csv_string_field(
    tmp_path, field, pii_value, description
):
    row = _seed_row(**{field: pii_value})
    data_root = _write_dataset(tmp_path, [row], create_images=False)

    with pytest.raises(InvalidReferenceCaseError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    message = str(error.value)
    assert "Potential PII" in message
    assert description in message
    assert field in message
    assert "line 2" in message
    assert pii_value not in message


def test_repository_rejects_absolute_image_paths(tmp_path):
    row = _seed_row(before_image_path="/Users/example/before.jpg")
    data_root = _write_dataset(tmp_path, [row])

    with pytest.raises(UnsafeImagePathError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert "absolute paths are not allowed" in str(error.value)


def test_repository_rejects_path_traversal_image_paths(tmp_path):
    row = _seed_row(after_image_path="images/DM-990/../after.jpg")
    data_root = _write_dataset(tmp_path, [row])

    with pytest.raises(UnsafeImagePathError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert "path traversal" in str(error.value)


@pytest.mark.parametrize("field", ["before_image_path", "after_image_path"])
def test_repository_rejects_missing_before_or_after_image_reference(tmp_path, field):
    row = _seed_row(**{field: ""})
    data_root = _write_dataset(tmp_path, [row])

    with pytest.raises(InvalidReferenceCaseError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert field in str(error.value)
    assert "exactly one value" in str(error.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("before_image_path", "images/DM-990/before.jpg;images/DM-990/before2.jpg"),
        ("after_image_path", "images/DM-990/after.jpg,images/DM-990/after2.jpg"),
        ("biologic", "Dupixent;Ebglyss"),
        ("outcome_label", "responder,partial_responder"),
    ],
)
def test_repository_rejects_multi_value_required_row_fields(tmp_path, field, value):
    row = _seed_row(**{field: value})
    data_root = _write_dataset(tmp_path, [row], create_images=False)

    with pytest.raises(InvalidReferenceCaseError) as error:
        ReferenceCaseRepository(data_root).list_cases()

    assert field in str(error.value)
    assert "multiple values" in str(error.value)


def test_real_reference_csv_passes_deidentification_path_and_completeness_checks():
    cases = ReferenceCaseRepository(DATA_ROOT).list_cases()

    assert len(cases) == 10
    for reference_case in cases:
        row = reference_case.row
        assert row.before_image_path
        assert row.after_image_path
        assert row.biologic in {"Dupixent", "Ebglyss"}
        assert row.outcome_label in {"responder", "partial_responder", "non_responder"}
        assert not Path(row.before_image_path).is_absolute()
        assert not Path(row.after_image_path).is_absolute()
        assert ".." not in Path(row.before_image_path).parts
        assert ".." not in Path(row.after_image_path).parts
