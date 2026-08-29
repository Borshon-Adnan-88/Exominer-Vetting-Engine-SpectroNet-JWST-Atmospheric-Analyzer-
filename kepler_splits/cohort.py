"""Load exactly the conflict-free Stage 1 binary cohort."""

from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import sha256_file
from .schemas import SplitValidationError

REQUIRED = {"source_row_index", "kepid", "kepoi_name", "label", "label_policy", "inclusion_status", "label_conflict", "koi_period", "koi_duration", "koi_model_snr"}


def load_cohort(config: dict) -> pd.DataFrame:
    path = Path(config["source_manifest"])
    actual = sha256_file(path)
    if actual != config["source_manifest_sha256"]:
        raise SplitValidationError(f"source manifest SHA-256 mismatch: {actual}")
    frame = pd.read_csv(path)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise SplitValidationError(f"missing source columns: {sorted(missing)}")
    cohort = frame.loc[(frame["inclusion_status"] == "included") & (~frame["label_conflict"].astype(bool))].copy()
    if cohort.empty or cohort[["kepid", "kepoi_name", "source_row_index", "label"]].isna().any().any():
        raise SplitValidationError("missing split identity or label")
    if set(cohort["label"].astype(float).unique()) != {0.0, 1.0}:
        raise SplitValidationError("cohort labels must be binary 0/1")
    if cohort["kepoi_name"].duplicated().any() or cohort["source_row_index"].duplicated().any():
        raise SplitValidationError("KOI identities must be unique")
    if cohort["label_policy"].nunique() != 1 or cohort["label_policy"].iat[0] != config["label_policy"]:
        raise SplitValidationError("label policy mismatch")
    cohort["kepid"] = cohort["kepid"].astype(np.int64)
    cohort["source_row_index"] = cohort["source_row_index"].astype(np.int64)
    cohort["label"] = cohort["label"].astype(np.int8)
    return cohort.sort_values(config["sort_fields"], kind="stable").reset_index(drop=True)
