"""Cross-field validation and cohort summaries."""

import pandas as pd

from .fields import LABEL_POLICY_NAME
from .schema import CatalogValidationError


def validate_manifest(frame: pd.DataFrame) -> None:
    required = {
        "source_row_index",
        "label",
        "label_name",
        "label_policy",
        "inclusion_status",
        "exclusion_reason",
        "label_conflict",
        "conflict_reason",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CatalogValidationError(f"manifest fields missing: {missing}")
    if not frame["label_policy"].eq(LABEL_POLICY_NAME).all():
        raise CatalogValidationError("manifest contains an unexpected label policy")
    included = frame["inclusion_status"].eq("included")
    if frame.loc[included, "label"].isna().any():
        raise CatalogValidationError("included rows must have labels")
    if frame.loc[~included, "label"].notna().any():
        raise CatalogValidationError("excluded rows must not have labels")
    if (frame.loc[included, "label_conflict"]).any():
        raise CatalogValidationError("conflicting rows cannot be included")


def summarize_manifest(frame: pd.DataFrame) -> dict[str, int | str]:
    host_sizes = frame.groupby("kepid", dropna=False).size()
    dispositions = frame["koi_disposition"].value_counts(dropna=False)
    return {
        "label_policy": LABEL_POLICY_NAME,
        "raw_row_count": int(len(frame)),
        "confirmed_count": int(dispositions.get("CONFIRMED", 0)),
        "false_positive_count": int(dispositions.get("FALSE POSITIVE", 0)),
        "candidate_count": int(dispositions.get("CANDIDATE", 0)),
        "unlabelled_count": int(
            dispositions.get("NOT DISPOSITIONED", 0) + frame["koi_disposition"].isna().sum()
        ),
        "conflict_count": int(frame["label_conflict"].sum()),
        "unique_kepid_hosts": int(frame["kepid"].nunique()),
        "multi_koi_hosts": int((host_sizes > 1).sum()),
        "included_binary_cohort_size": int(frame["inclusion_status"].eq("included").sum()),
    }
