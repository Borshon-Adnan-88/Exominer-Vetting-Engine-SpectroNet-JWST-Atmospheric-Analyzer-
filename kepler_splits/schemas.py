"""Stage 3 schema constants and validation errors."""

ASSIGNMENT_COLUMNS = [
    "split_policy_version", "source_manifest_sha256", "label_policy", "strategy",
    "repeat_id", "seed", "fold_id", "partition", "source_row_index",
    "kepoi_name", "kepid", "label",
]
PARTITION_ORDER = {"train": 0, "validation": 1, "test": 2}


class SplitValidationError(ValueError):
    """Raised when the frozen cohort or a split violates policy."""
