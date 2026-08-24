"""Offline schema normalization and validation."""

from pathlib import Path

import numpy as np
import pandas as pd

from .fields import INTEGER_FIELDS, NUMERIC_FIELDS, SELECTED_FIELDS, STRING_FIELDS


class CatalogValidationError(ValueError):
    """Raised when a catalogue violates the frozen Stage 1 contract."""


def read_catalog(path: str | Path) -> pd.DataFrame:
    """Read and validate a selected-field DR25 KOI CSV snapshot."""
    frame = pd.read_csv(path, dtype={field: "string" for field in STRING_FIELDS})
    return normalize_and_validate(frame)


def normalize_and_validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(SELECTED_FIELDS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(SELECTED_FIELDS))
    if missing or unexpected:
        raise CatalogValidationError(
            f"catalog columns differ from schema; missing={missing}, unexpected={unexpected}"
        )

    result = frame.loc[:, SELECTED_FIELDS].copy()
    for field in STRING_FIELDS:
        result[field] = result[field].astype("string").str.strip().replace("", pd.NA)

    for field in INTEGER_FIELDS:
        numeric = pd.to_numeric(result[field], errors="coerce")
        invalid = result[field].notna() & numeric.isna()
        fractional = numeric.notna() & (numeric % 1 != 0)
        if invalid.any() or fractional.any():
            raise CatalogValidationError(f"{field} contains invalid integer values")
        result[field] = numeric.astype("Int64")

    for field in NUMERIC_FIELDS:
        numeric = pd.to_numeric(result[field], errors="coerce")
        invalid = result[field].notna() & numeric.isna()
        if invalid.any() or np.isinf(numeric.dropna().to_numpy(dtype=float)).any():
            raise CatalogValidationError(f"{field} contains invalid or non-finite values")
        result[field] = numeric.astype("Float64")

    for field in ("kepid", "kepoi_name", "koi_disposition"):
        if result[field].isna().any():
            raise CatalogValidationError(f"{field} must not be missing")

    if result["kepoi_name"].duplicated().any():
        duplicates = result.loc[result["kepoi_name"].duplicated(False), "kepoi_name"].tolist()
        raise CatalogValidationError(f"kepoi_name must be unique; duplicates={duplicates[:5]}")

    allowed = {"CONFIRMED", "FALSE POSITIVE", "CANDIDATE", "NOT DISPOSITIONED"}
    unknown = sorted(set(result["koi_disposition"].dropna()) - allowed)
    if unknown:
        raise CatalogValidationError(f"unknown koi_disposition values: {unknown}")

    project_allowed = {"CANDIDATE", "FALSE POSITIVE", "NOT DISPOSITIONED"}
    project_unknown = sorted(set(result["koi_pdisposition"].dropna()) - project_allowed)
    if project_unknown:
        raise CatalogValidationError(f"unknown koi_pdisposition values: {project_unknown}")

    score = result["koi_score"].dropna()
    if ((score < 0) | (score > 1)).any():
        raise CatalogValidationError("koi_score must be between 0 and 1 when present")

    for flag in ("koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec"):
        if not result[flag].dropna().isin([0, 1]).all():
            raise CatalogValidationError(f"{flag} must contain only 0, 1, or null")

    federation = result.dropna(subset=["koi_tce_plnt_num", "koi_tce_delivname"])
    key = ["kepid", "koi_tce_plnt_num", "koi_tce_delivname"]
    if federation.duplicated(key).any():
        raise CatalogValidationError("federated TCE key must be unique when complete")

    return result
