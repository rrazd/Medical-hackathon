"""Loader for the simple 4-column reference dataset.

The curated dataset has only four columns: ``Before``, ``After``, ``Biologic``,
``Age``. ``Before``/``After`` are local image paths (relative to the data root).
Because there is no explicit outcome column, treatment response for each case is
derived by comparing biomarkers extracted from the Before image against the After
image: a larger visible improvement means a stronger response.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from app.schemas.predict import PatientFeatures
from app.services.biomarker_extraction import extract_biomarkers

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CSV_NAME = "cases.csv"

CANONICAL_BIOLOGICS = {"dupixent": "Dupixent", "ebglyss": "Ebglyss"}
REQUIRED_COLUMNS = ("before", "after", "biologic", "age")

# How much each biomarker's Before->After reduction contributes to the response
# (improvement) score. Percent fields are normalized to 0..1 before weighting.
IMPROVEMENT_WEIGHTS: Dict[str, float] = {
    "erythema_score": 0.30,
    "inflammation_score": 0.30,
    "lesion_coverage_pct": 0.20,
    "texture_score": 0.10,
    "dryness_scaling_score": 0.10,
}
# Raw weighted Before->After delta that maps to a full improvement score of 1.0.
IMPROVEMENT_SCALE = 0.5

RESPONDER_THRESHOLD = 0.66
PARTIAL_THRESHOLD = 0.33


class ImageDatasetError(Exception):
    """Raised when the 4-column dataset cannot be loaded or validated."""


@dataclass(frozen=True)
class ImageReferenceCase:
    case_id: str
    biologic: str
    age: int
    before_path: str
    after_path: str
    before_features: PatientFeatures
    after_features: PatientFeatures
    improvement_score: float
    outcome_label: str


def _clip_unit(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _normalized(features: PatientFeatures, field: str) -> float:
    value = float(getattr(features, field))
    return value / 100.0 if field.endswith("_pct") else value


def compute_improvement_score(
    before: PatientFeatures, after: PatientFeatures
) -> float:
    """Derive a 0..1 response score from the Before->After biomarker reduction."""
    raw = 0.0
    for field, weight in IMPROVEMENT_WEIGHTS.items():
        raw += weight * (_normalized(before, field) - _normalized(after, field))
    return _clip_unit(raw / IMPROVEMENT_SCALE)


def _outcome_label(improvement_score: float) -> str:
    if improvement_score >= RESPONDER_THRESHOLD:
        return "responder"
    if improvement_score >= PARTIAL_THRESHOLD:
        return "partial_responder"
    return "non_responder"


class ImageReferenceRepository:
    def __init__(
        self,
        data_root: Optional[Path] = None,
        csv_path: Optional[Path] = None,
    ) -> None:
        self.data_root = (data_root or DEFAULT_DATA_ROOT).resolve()
        self.csv_path = (
            Path(csv_path).resolve()
            if csv_path is not None
            else self.data_root / DEFAULT_CSV_NAME
        )
        self._cases: Optional[List[ImageReferenceCase]] = None

    def list_cases(self) -> List[ImageReferenceCase]:
        if self._cases is None:
            self._cases = self._load()
        return self._cases

    def _resolve_image(self, relative_path: str, row_number: int) -> Path:
        if not relative_path:
            raise ImageDatasetError(f"Row {row_number}: image path is empty.")
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ImageDatasetError(
                f"Row {row_number}: image path must be relative, got '{relative_path}'."
            )
        resolved = (self.data_root / candidate).resolve()
        if not resolved.is_relative_to(self.data_root):
            raise ImageDatasetError(
                f"Row {row_number}: image path escapes the data root: '{relative_path}'."
            )
        if not resolved.is_file():
            raise ImageDatasetError(
                f"Row {row_number}: image file not found: '{relative_path}'."
            )
        return resolved

    def _load(self) -> List[ImageReferenceCase]:
        if not self.csv_path.is_file():
            raise ImageDatasetError(f"Dataset CSV not found: {self.csv_path}")

        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            header_map = {name.lower().strip(): name for name in fieldnames}
            missing = [col for col in REQUIRED_COLUMNS if col not in header_map]
            if missing:
                raise ImageDatasetError(
                    "Dataset CSV is missing required column(s): "
                    + ", ".join(missing)
                    + f". Expected columns: {', '.join(c.title() for c in REQUIRED_COLUMNS)}."
                )

            cases: List[ImageReferenceCase] = []
            for index, row in enumerate(reader):
                row_number = index + 2  # account for header line
                before_rel = (row.get(header_map["before"]) or "").strip()
                after_rel = (row.get(header_map["after"]) or "").strip()
                biologic_raw = (row.get(header_map["biologic"]) or "").strip()
                age_raw = (row.get(header_map["age"]) or "").strip()

                biologic_key = biologic_raw.lower()
                if biologic_key not in CANONICAL_BIOLOGICS:
                    raise ImageDatasetError(
                        f"Row {row_number}: biologic must be Dupixent or Ebglyss, "
                        f"got '{biologic_raw}'."
                    )
                try:
                    age = int(float(age_raw))
                except ValueError as exc:
                    raise ImageDatasetError(
                        f"Row {row_number}: age must be a number, got '{age_raw}'."
                    ) from exc
                if not 1 <= age <= 129:
                    raise ImageDatasetError(
                        f"Row {row_number}: age must be between 1 and 129, got {age}."
                    )

                before_file = self._resolve_image(before_rel, row_number)
                after_file = self._resolve_image(after_rel, row_number)

                before_features, _ = extract_biomarkers(before_file.read_bytes())
                after_features, _ = extract_biomarkers(after_file.read_bytes())
                improvement = compute_improvement_score(before_features, after_features)

                cases.append(
                    ImageReferenceCase(
                        case_id=f"C{index + 1:03d}",
                        biologic=CANONICAL_BIOLOGICS[biologic_key],
                        age=age,
                        before_path=before_rel,
                        after_path=after_rel,
                        before_features=before_features,
                        after_features=after_features,
                        improvement_score=improvement,
                        outcome_label=_outcome_label(improvement),
                    )
                )

            if not cases:
                raise ImageDatasetError("Dataset CSV contains no rows.")
            return cases
