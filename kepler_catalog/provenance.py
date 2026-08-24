"""Provenance records and deterministic checksum helpers."""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .download import METADATA_QUERY, TAP_ENDPOINT
from .fields import DATASET_ID, LABEL_POLICY_NAME, SCHEMA_VERSION, SELECTED_FIELDS, SOURCE_TABLE


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_provenance(
    *, raw_path: str | Path, raw_sha256: str, raw_bytes: int,
    raw_rows: int, query: str, tap_schema_sha256: str,
    tap_schema_column_count: int, code_revision: str | None,
) -> dict[str, object]:
    return {
        "dataset_id": DATASET_ID,
        "archive": "NASA Exoplanet Archive",
        "archive_doi": "10.26133/NEA4",
        "service": "TAP",
        "endpoint": TAP_ENDPOINT,
        "table": SOURCE_TABLE,
        "release": "Q1-Q17 DR25",
        "pipeline_release": "SOC 9.3",
        "catalog_status": "DONE",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "query_adql": query,
        "tap_metadata_query_adql": METADATA_QUERY,
        "selected_fields": list(SELECTED_FIELDS),
        "raw_filename": Path(raw_path).name,
        "raw_bytes": raw_bytes,
        "raw_rows": raw_rows,
        "raw_columns": len(SELECTED_FIELDS),
        "raw_sha256": raw_sha256,
        "tap_schema_sha256": tap_schema_sha256,
        "tap_schema_column_count": tap_schema_column_count,
        "schema_version": SCHEMA_VERSION,
        "label_policy": LABEL_POLICY_NAME,
        "code_revision": code_revision,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def write_json(data: dict[str, object], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
