import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

from pydantic import ValidationError

from app.schemas.reference_case import BIOMARKER_FIELDS, ReferenceCase, ReferenceCaseRow


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
REQUIRED_COLUMNS = tuple(ReferenceCaseRow.model_fields.keys())
SINGLE_VALUE_COLUMNS = ("before_image_path", "after_image_path", "biologic", "outcome_label")
PII_PATTERNS: Tuple[Tuple[str, Pattern[str]], ...] = (
    ("SSN-like value", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "phone-like value",
        re.compile(
            r"(?:\b\d{3}[-.]\d{3}[-.]\d{4}\b|\(\d{3}\)\s*\d{3}[-.]\d{4}\b|\b\d{10}\b)"
        ),
    ),
    (
        "email address",
        re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
    ),
    ("MRN-like value", re.compile(r"\bMRN[:#\s-]*[A-Z0-9-]*\b", re.IGNORECASE)),
    ("PII label", re.compile(r"\b(?:MRN|DOB|SSN|patient|name)\b", re.IGNORECASE)),
    ("date-like value", re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")),
    ("likely personal name", re.compile(r"\b[A-Z][a-z]{1,}\s+[A-Z][a-z]{1,}\b")),
)


class ReferenceDatasetError(Exception):
    """Base class for reference dataset loading failures."""


class ReferenceDatasetSchemaError(ReferenceDatasetError):
    """Raised when the CSV header does not match the required schema."""


class InvalidReferenceCaseError(ReferenceDatasetError):
    """Raised when a CSV row cannot be parsed into a valid reference case."""


class UnsafeImagePathError(ReferenceDatasetError):
    """Raised when an image path is absolute, escaping, mis-scoped, or ambiguous."""


class MissingImageFileError(ReferenceDatasetError):
    """Raised when a resolved image path does not exist as a file."""


class ReferenceCaseRepository:
    def __init__(self, data_root: Optional[Path] = None) -> None:
        self.data_root = (data_root or DEFAULT_DATA_ROOT).resolve()
        self.csv_path = self.data_root / "reference_cases.csv"
        self._cases: Optional[List[ReferenceCase]] = None
        self._feature_vectors: Optional[Dict[str, List[float]]] = None

    def list_cases(self) -> List[ReferenceCase]:
        self._ensure_loaded()
        return list(self._cases or [])

    def feature_vectors_by_case_id(self) -> Dict[str, List[float]]:
        self._ensure_loaded()
        return {case_id: list(vector) for case_id, vector in (self._feature_vectors or {}).items()}

    def _ensure_loaded(self) -> None:
        if self._cases is not None and self._feature_vectors is not None:
            return
        cases = self._load_cases()
        self._cases = cases
        self._feature_vectors = {
            reference_case.row.case_id: reference_case.row.biomarker_vector()
            for reference_case in cases
        }

    def _load_cases(self) -> List[ReferenceCase]:
        try:
            with self.csv_path.open(newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                self._validate_columns(reader.fieldnames)
                return self._read_rows(reader)
        except FileNotFoundError as exc:
            raise ReferenceDatasetSchemaError(f"Reference dataset CSV not found: {self.csv_path}") from exc

    def _validate_columns(self, fieldnames: Optional[Sequence[str]]) -> None:
        if fieldnames is None:
            raise ReferenceDatasetSchemaError("Reference dataset CSV is missing a header row")
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        unexpected = [column for column in fieldnames if column not in REQUIRED_COLUMNS]
        if missing or unexpected:
            details = []
            if missing:
                details.append("missing columns: " + ", ".join(missing))
            if unexpected:
                details.append("unexpected columns: " + ", ".join(unexpected))
            raise ReferenceDatasetSchemaError("Invalid reference dataset schema: " + "; ".join(details))
        if list(fieldnames) != list(REQUIRED_COLUMNS):
            raise ReferenceDatasetSchemaError(
                "Invalid reference dataset schema: columns must be ordered as "
                + ", ".join(REQUIRED_COLUMNS)
            )

    def _read_rows(self, reader: csv.DictReader) -> List[ReferenceCase]:
        cases: List[ReferenceCase] = []
        seen_case_ids = set()
        for row_number, raw_row in enumerate(reader, start=2):
            row = self._validate_row(raw_row, row_number)
            if row.case_id in seen_case_ids:
                raise InvalidReferenceCaseError(
                    f"Duplicate reference case_id {row.case_id} at line {row_number}"
                )
            seen_case_ids.add(row.case_id)
            before_file = self._resolve_image_path(row.before_image_path, row.case_id, "before_image_path")
            after_file = self._resolve_image_path(row.after_image_path, row.case_id, "after_image_path")
            if before_file == after_file:
                raise UnsafeImagePathError(
                    f"Reference case {row.case_id} before_image_path and after_image_path resolve to the same file"
                )
            cases.append(
                ReferenceCase(row=row, before_image_file=before_file, after_image_file=after_file)
            )
        return cases

    def _validate_row(self, raw_row: Dict[str, Optional[str]], row_number: int) -> ReferenceCaseRow:
        context_case_id = raw_row.get("case_id") or "<missing case_id>"
        try:
            self._assert_no_pii(raw_row, row_number)
            self._assert_single_required_values(raw_row, row_number)
            return ReferenceCaseRow.model_validate(raw_row)
        except ValidationError as exc:
            fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
            field_details = ", ".join(fields) if fields else "row"
            raise InvalidReferenceCaseError(
                f"Invalid reference case at line {row_number} ({context_case_id}): {field_details}: {exc}"
            ) from exc

    def _assert_no_pii(self, raw_row: Dict[str, Optional[str]], row_number: int) -> None:
        for column, value in raw_row.items():
            text = str(value or "")
            if not text:
                continue
            for description, pattern in PII_PATTERNS:
                if pattern.search(text):
                    raise InvalidReferenceCaseError(
                        f"Potential PII ({description}) in column {column} at line {row_number}"
                    )

    def _assert_single_required_values(
        self, raw_row: Dict[str, Optional[str]], row_number: int
    ) -> None:
        for column in SINGLE_VALUE_COLUMNS:
            value = raw_row.get(column)
            text = (value or "").strip()
            if not text:
                raise InvalidReferenceCaseError(
                    f"Reference dataset column {column} must contain exactly one value at line {row_number}"
                )
            if "," in text or ";" in text:
                raise InvalidReferenceCaseError(
                    f"Reference dataset column {column} must not contain multiple values at line {row_number}"
                )

    def _resolve_image_path(self, image_path: str, case_id: str, field_name: str) -> Path:
        relative_path = Path(image_path)
        if not image_path.strip():
            raise UnsafeImagePathError(f"Reference case {case_id} {field_name} is empty")
        if relative_path.is_absolute():
            raise UnsafeImagePathError(
                f"Reference case {case_id} {field_name} must be relative; absolute paths are not allowed"
            )
        if ".." in relative_path.parts:
            raise UnsafeImagePathError(
                f"Reference case {case_id} {field_name} contains path traversal"
            )
        expected_prefix = ("images", case_id)
        if len(relative_path.parts) < 3 or relative_path.parts[:2] != expected_prefix:
            raise UnsafeImagePathError(
                f"Reference case {case_id} {field_name} must be scoped under images/{case_id}/"
            )
        resolved = (self.data_root / relative_path).resolve()
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise UnsafeImagePathError(
                f"Reference case {case_id} {field_name} resolves outside the data root"
            ) from exc
        if not resolved.is_file():
            raise MissingImageFileError(
                f"Reference case {case_id} {field_name} file is missing: {relative_path.as_posix()}"
            )
        return resolved


__all__ = [
    "BIOMARKER_FIELDS",
    "DEFAULT_DATA_ROOT",
    "REQUIRED_COLUMNS",
    "InvalidReferenceCaseError",
    "MissingImageFileError",
    "ReferenceCaseRepository",
    "ReferenceDatasetError",
    "ReferenceDatasetSchemaError",
    "UnsafeImagePathError",
]
