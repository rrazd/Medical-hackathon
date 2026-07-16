import csv
from pathlib import Path

from app.schemas.predict import PatientFeatures


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

BIOMARKER_FIELDS = [
    "erythema_score",
    "lesion_coverage_pct",
    "texture_score",
    "dryness_scaling_score",
    "inflammation_score",
    "affected_body_area_pct",
]


def _read_rows():
    with CSV_PATH.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return reader.fieldnames, list(reader)


def test_reference_csv_uses_canonical_header():
    header, _ = _read_rows()

    assert header == CANONICAL_HEADER
    assert list(PatientFeatures.model_fields) == BIOMARKER_FIELDS


def test_reference_csv_has_required_seed_distribution():
    _, rows = _read_rows()

    assert len(rows) == 10
    assert [row["case_id"] for row in rows] == [f"DM-{index:03d}" for index in range(1, 11)]
    assert sum(row["biologic"] == "Dupixent" for row in rows) == 5
    assert sum(row["biologic"] == "Ebglyss" for row in rows) == 5
    assert {row["fitzpatrick_skin_type"] for row in rows} == {"I", "II", "III", "IV", "V", "VI"}
    assert sum(row["fitzpatrick_skin_type"] in {"IV", "V", "VI"} for row in rows) >= 3


def test_reference_csv_rows_have_required_values():
    _, rows = _read_rows()
    outcome_scores = {"responder": "1.0", "partial_responder": "0.5", "non_responder": "0.0"}

    assert sum(row["outcome_label"] == "responder" for row in rows) >= 3
    assert sum(row["outcome_label"] == "partial_responder" for row in rows) >= 4
    assert sum(row["outcome_label"] == "non_responder" for row in rows) >= 3

    for row in rows:
        assert row["biologic"]
        assert row["outcome_label"] in outcome_scores
        assert row["outcome_score"] == outcome_scores[row["outcome_label"]]
        assert row["before_image_path"]
        assert row["after_image_path"]
        for field in BIOMARKER_FIELDS:
            assert row[field] != ""
            assert float(row[field]) >= 0.0


def test_reference_csv_image_paths_are_relative_case_scoped_jpegs():
    _, rows = _read_rows()

    for row in rows:
        before_path = Path(row["before_image_path"])
        after_path = Path(row["after_image_path"])

        assert not before_path.is_absolute()
        assert not after_path.is_absolute()
        assert ".." not in before_path.parts
        assert ".." not in after_path.parts
        assert row["before_image_path"].startswith(f"images/{row['case_id']}/")
        assert row["after_image_path"].startswith(f"images/{row['case_id']}/")
        assert before_path != after_path
        assert before_path.name == "before.jpg"
        assert after_path.name == "after.jpg"
        assert (DATA_ROOT / before_path).is_file()
        assert (DATA_ROOT / after_path).is_file()
