"""Deterministic Stage 1 manifest construction."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from .labels import apply_label_policy
from .schema import read_catalog
from .validate import summarize_manifest, validate_manifest


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n", float_format="%.15g").encode("utf-8")


def build_manifest(raw_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    raw = read_catalog(raw_path)
    raw = raw.sort_values(["kepid", "kepoi_name"], kind="stable").reset_index(drop=True)
    raw.insert(0, "source_row_index", range(len(raw)))
    manifest = apply_label_policy(raw)
    validate_manifest(manifest)
    conflicts = manifest.loc[manifest["label_conflict"]].copy()
    return manifest, conflicts, summarize_manifest(manifest)


def write_manifest_artifacts(
    manifest: pd.DataFrame,
    conflicts: pd.DataFrame,
    summary: dict[str, object],
    manifest_path: str | Path,
    conflict_path: str | Path,
    summary_path: str | Path,
) -> dict[str, str]:
    manifest_path = Path(manifest_path)
    conflict_path = Path(conflict_path)
    summary_path = Path(summary_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = _csv_bytes(manifest)
    conflict_bytes = _csv_bytes(conflicts)
    manifest_path.write_bytes(manifest_bytes)
    conflict_path.write_bytes(conflict_bytes)
    hashes = {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "conflicts_sha256": hashlib.sha256(conflict_bytes).hexdigest(),
    }
    final_summary = {**summary, **hashes}
    summary_path.write_text(
        json.dumps(final_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hashes
