"""NASA Exoplanet Archive TAP schema validation and snapshot retrieval."""

import csv
import hashlib
import io
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .fields import SELECTED_FIELDS, SOURCE_TABLE

TAP_ENDPOINT = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
METADATA_QUERY = (
    "SELECT column_name FROM TAP_SCHEMA.columns "
    f"WHERE table_name = '{SOURCE_TABLE.upper()}' ORDER BY column_name"
)


class TapSchemaError(RuntimeError):
    """Raised before retrieval when the live TAP schema violates expectations."""


def _tap_request(query: str, timeout: int = 120) -> bytes:
    payload = urlencode({"query": query, "format": "csv"}).encode("utf-8")
    request = Request(TAP_ENDPOINT, data=payload, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    request.add_header("User-Agent", "exominer-research-stage1/0.1")
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"TAP request failed with HTTP {response.status}")
        return response.read()


def get_live_columns(timeout: int = 120) -> tuple[set[str], bytes]:
    metadata_bytes = _tap_request(METADATA_QUERY, timeout=timeout)
    text = metadata_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise TapSchemaError("TAP metadata response has no header")
    key = next((name for name in reader.fieldnames if name.lower() == "column_name"), None)
    if key is None:
        raise TapSchemaError(f"TAP metadata lacks column_name; got {reader.fieldnames}")
    return {row[key].strip().lower() for row in reader if row.get(key)}, metadata_bytes


def validate_live_schema(timeout: int = 120) -> tuple[set[str], bytes]:
    columns, metadata_bytes = get_live_columns(timeout=timeout)
    missing = sorted(set(SELECTED_FIELDS) - columns)
    if missing:
        raise TapSchemaError(
            f"{SOURCE_TABLE} is missing requested fields: {missing}; retrieval aborted"
        )
    return columns, metadata_bytes


def retrieve_snapshot(query: str, raw_directory: str | Path, timeout: int = 300) -> dict[str, object]:
    """Validate metadata, hash response bytes, and save an immutable raw snapshot."""
    columns, metadata_bytes = validate_live_schema(timeout=timeout)
    raw_bytes = _tap_request(query, timeout=timeout)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    metadata_digest = hashlib.sha256(metadata_bytes).hexdigest()
    destination_dir = Path(raw_directory)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"kepler_q1_q17_dr25_koi_{digest[:16]}.csv"
    if destination.exists() and destination.read_bytes() != raw_bytes:
        raise RuntimeError(f"refusing to overwrite non-identical snapshot: {destination}")
    if not destination.exists():
        destination.write_bytes(raw_bytes)
    return {
        "path": destination,
        "sha256": digest,
        "bytes": len(raw_bytes),
        "tap_schema_sha256": metadata_digest,
        "tap_schema_column_count": len(columns),
    }
