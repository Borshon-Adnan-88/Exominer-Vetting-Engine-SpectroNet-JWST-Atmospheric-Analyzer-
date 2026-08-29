"""Matched 20-fold object and host-group assignments."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .schemas import ASSIGNMENT_COLUMNS, PARTITION_ORDER, SplitValidationError


def _partition_map(config: dict) -> dict[int, str]:
    mapping = {}
    for partition, folds in config["algorithm"]["fold_partition_map"].items():
        for fold in folds:
            if fold in mapping:
                raise SplitValidationError("fold appears in multiple partitions")
            mapping[int(fold)] = partition
    if set(mapping) != set(range(config["algorithm"]["n_splits"])):
        raise SplitValidationError("fold-to-partition map is incomplete")
    return mapping


def _folds(cohort: pd.DataFrame, strategy: str, seed: int, config: dict) -> np.ndarray:
    n = config["algorithm"]["n_splits"]
    splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed) if strategy == "object_stratified" else StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    groups = None if strategy == "object_stratified" else cohort["kepid"].to_numpy()
    result = np.full(len(cohort), -1, dtype=np.int16)
    for fold_id, (_, test_indices) in enumerate(splitter.split(np.zeros(len(cohort)), cohort["label"], groups)):
        result[test_indices] = fold_id
    if (result < 0).any():
        raise SplitValidationError("incomplete fold assignment")
    return result


def build_assignments(cohort: pd.DataFrame, config: dict) -> pd.DataFrame:
    mapping = _partition_map(config); records = []
    identity = cohort[["source_row_index", "kepoi_name", "kepid", "label"]]
    for strategy in config["strategies"]:
        for repeat_id, seed in enumerate(config["seeds"]):
            block = identity.copy(); block["fold_id"] = _folds(cohort, strategy, seed, config)
            block["partition"] = block["fold_id"].map(mapping)
            block["split_policy_version"] = config["split_policy_version"]
            block["source_manifest_sha256"] = config["source_manifest_sha256"]
            block["label_policy"] = config["label_policy"]
            block["strategy"] = strategy; block["repeat_id"] = repeat_id; block["seed"] = seed
            records.append(block[ASSIGNMENT_COLUMNS])
    result = pd.concat(records, ignore_index=True)
    result["_partition_order"] = result["partition"].map(PARTITION_ORDER)
    return result.sort_values(["strategy", "repeat_id", "_partition_order", "kepid", "kepoi_name"], kind="stable").drop(columns="_partition_order").reset_index(drop=True)
