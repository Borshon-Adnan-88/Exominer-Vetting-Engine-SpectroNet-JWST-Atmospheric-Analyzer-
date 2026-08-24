"""Versioned binary label policy and conflict auditing."""

import pandas as pd

from .fields import LABEL_POLICY_NAME
from .schema import CatalogValidationError


def _conflict_reason(row: pd.Series) -> str | None:
    disposition = row["koi_disposition"]
    project = row["koi_pdisposition"]
    has_name = pd.notna(row["kepler_name"])
    if disposition == "CONFIRMED" and project == "FALSE POSITIVE":
        return "confirmed_but_robovetter_false_positive"
    if disposition == "CONFIRMED" and not has_name:
        return "confirmed_without_kepler_name"
    if disposition != "CONFIRMED" and has_name:
        return "nonconfirmed_with_kepler_name"
    return None


def apply_label_policy(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply confirmed_vs_false_positive_v1 without dropping audit rows."""
    result = frame.copy()
    reasons = result.apply(_conflict_reason, axis=1).astype("string")
    result["conflict_reason"] = reasons
    result["label_conflict"] = reasons.notna()
    result["label"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["label_name"] = pd.Series(pd.NA, index=result.index, dtype="string")
    result["label_policy"] = LABEL_POLICY_NAME
    result["inclusion_status"] = "excluded_unlabelled"
    result["exclusion_reason"] = "unlabelled_or_not_dispositioned"

    confirmed = result["koi_disposition"].eq("CONFIRMED")
    false_positive = result["koi_disposition"].eq("FALSE POSITIVE")
    candidate = result["koi_disposition"].eq("CANDIDATE")
    conflict = result["label_conflict"]

    result.loc[candidate, ["inclusion_status", "exclusion_reason"]] = [
        "excluded_candidate",
        "candidate_excluded_by_policy",
    ]
    result.loc[conflict, ["inclusion_status", "exclusion_reason"]] = [
        "excluded_conflict",
        "contradictory_source_labels",
    ]

    included_confirmed = confirmed & ~conflict
    included_fp = false_positive & ~conflict
    result.loc[included_confirmed, "label"] = 1
    result.loc[included_confirmed, "label_name"] = "positive_confirmed"
    result.loc[included_fp, "label"] = 0
    result.loc[included_fp, "label_name"] = "negative_false_positive"
    result.loc[included_confirmed | included_fp, "inclusion_status"] = "included"
    result.loc[included_confirmed | included_fp, "exclusion_reason"] = pd.NA

    included = result["inclusion_status"].eq("included")
    for field in ("koi_period", "koi_time0bk", "koi_duration"):
        if result.loc[included, field].isna().any():
            raise CatalogValidationError(f"included rows must have non-null {field}")
        if (result.loc[included, field] <= 0).any():
            raise CatalogValidationError(f"included rows must have positive {field}")
    return result
