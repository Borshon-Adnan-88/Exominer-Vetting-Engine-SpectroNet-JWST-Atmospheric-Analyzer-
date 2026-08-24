"""Validate the live TAP schema and retrieve the frozen Stage 1 KOI snapshot."""

import argparse
import subprocess
from pathlib import Path

from kepler_catalog.download import retrieve_snapshot
from kepler_catalog.provenance import build_provenance, write_json
from kepler_catalog.schema import read_catalog


def _revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=Path, default=Path("queries/kepler_q1_q17_dr25_koi.adql"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--provenance", type=Path, default=Path("data/provenance/kepler_q1_q17_dr25_koi.json"))
    args = parser.parse_args()
    query = args.query.read_text(encoding="utf-8").strip()
    result = retrieve_snapshot(query, args.raw_dir)
    frame = read_catalog(result["path"])
    provenance = build_provenance(
        raw_path=result["path"], raw_sha256=result["sha256"], raw_bytes=result["bytes"],
        raw_rows=len(frame), query=query, tap_schema_sha256=result["tap_schema_sha256"],
        tap_schema_column_count=result["tap_schema_column_count"], code_revision=_revision(),
    )
    write_json(provenance, args.provenance)
    print(f"Retrieved {len(frame)} rows to {result['path']}")
    print(f"Raw SHA-256: {result['sha256']}")


if __name__ == "__main__":
    main()
