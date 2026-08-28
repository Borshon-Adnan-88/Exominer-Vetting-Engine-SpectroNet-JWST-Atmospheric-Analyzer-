"""Deterministic Stage 2 pilot selection from the frozen Stage 1 manifest."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _period_stratum(period: float) -> str:
    if period < 3:
        return "very_short"
    if period < 10:
        return "short"
    if period < 100:
        return "medium"
    return "long"


def select_pilot(manifest_path: str | Path, config_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    actual_hash = sha256_file(manifest_path)
    if actual_hash != config["source"]["manifest_sha256"]:
        raise ValueError("Stage 1 manifest checksum mismatch")
    full = pd.read_csv(manifest_path)
    host_counts = full.groupby("kepid").size()
    eligible = full.loc[(full["inclusion_status"] == "included") & (~full["label_conflict"].astype(bool))].copy()
    finite = config["eligibility"]["require_finite_fields"]
    positive = config["eligibility"]["require_positive_fields"]
    for field in finite:
        eligible = eligible.loc[np.isfinite(pd.to_numeric(eligible[field], errors="coerce"))]
    for field in positive:
        eligible = eligible.loc[pd.to_numeric(eligible[field], errors="coerce") > 0]
    eligible = eligible.loc[eligible["koi_duration"] < eligible["koi_period"] * 24]
    eligible["host_manifest_count"] = eligible["kepid"].map(host_counts).astype(int)
    eligible["host_category"] = np.where(eligible["host_manifest_count"] >= 2, "multi", "single")
    eligible["period_stratum"] = eligible["koi_period"].map(_period_stratum)
    eligible["selected"] = False
    eligible["cell_rank"] = pd.Series(pd.NA, index=eligible.index, dtype="Int64")
    eligible["snr_distance"] = np.nan
    eligible["duration_distance"] = np.nan

    selected_indices: list[int] = []
    used_hosts: set[int] = set()
    for label in config["stratification"]["labels"]:
        for host in config["stratification"]["host_categories"]:
            for stratum in config["stratification"]["period_strata"]:
                mask = (eligible["label"] == label) & (eligible["host_category"] == host) & (eligible["period_stratum"] == stratum)
                cell = eligible.loc[mask].copy()
                if cell.empty:
                    raise ValueError(f"empty pilot cell: label={label}, host={host}, period={stratum}")
                snr_pct = np.log10(cell["koi_model_snr"]).rank(method="average", pct=True)
                dur_pct = np.log10(cell["koi_duration"]).rank(method="average", pct=True)
                cell["snr_distance"] = (snr_pct - config["ranking"]["snr_target_percentiles"][stratum]).abs()
                cell["duration_distance"] = (dur_pct - config["ranking"]["duration_target_percentiles"][stratum]).abs()
                cell = cell.sort_values(config["ranking"]["sort_keys"], kind="stable")
                eligible.loc[cell.index, "snr_distance"] = cell["snr_distance"]
                eligible.loc[cell.index, "duration_distance"] = cell["duration_distance"]
                eligible.loc[cell.index, "cell_rank"] = range(1, len(cell) + 1)
                available = cell.loc[~cell["kepid"].astype(int).isin(used_hosts)]
                if available.empty:
                    raise ValueError(f"no unique host remains for pilot cell {label}/{host}/{stratum}")
                chosen = int(available.index[0])
                selected_indices.append(chosen)
                used_hosts.add(int(eligible.loc[chosen, "kepid"]))
    eligible.loc[selected_indices, "selected"] = True
    ranking = eligible.sort_values(["label", "host_category", "period_stratum", "cell_rank", "kepid"], kind="stable")
    selected = eligible.loc[selected_indices].copy().reset_index(drop=True)
    selected.insert(0, "pilot_order", range(1, len(selected) + 1))
    if len(selected) != config["selection_size"] or selected["kepid"].duplicated().any():
        raise ValueError("pilot selection violates size or unique-host contract")
    summary = {
        "pilot_id": config["pilot_id"], "source_manifest_sha256": actual_hash,
        "eligible_count": int(len(eligible)), "selected_count": int(len(selected)),
        "unique_selected_hosts": int(selected["kepid"].nunique()),
    }
    return ranking.reset_index(drop=True), selected, summary
