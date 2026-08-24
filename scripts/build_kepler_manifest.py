"""Build deterministic Stage 1 manifests from an existing raw snapshot."""

import argparse
import json
from pathlib import Path

from kepler_catalog.manifest import build_manifest, write_manifest_artifacts
from kepler_catalog.provenance import sha256_file, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance", type=Path, default=Path("data/provenance/kepler_q1_q17_dr25_koi.json"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--checksums", type=Path, default=Path("data/provenance/checksums.sha256"))
    args = parser.parse_args()

    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    raw_path = args.raw_dir / provenance["raw_filename"]
    actual_raw_hash = sha256_file(raw_path)
    if actual_raw_hash != provenance["raw_sha256"]:
        raise RuntimeError("raw snapshot checksum mismatch; manifest build aborted")

    manifest, conflicts, summary = build_manifest(raw_path)
    manifest_path = args.output_dir / "kepler_q1_q17_dr25_koi_manifest.csv"
    conflict_path = args.output_dir / "kepler_q1_q17_dr25_koi_conflicts.csv"
    summary_path = args.output_dir / "kepler_q1_q17_dr25_koi_summary.json"
    hashes = write_manifest_artifacts(manifest, conflicts, summary, manifest_path, conflict_path, summary_path)
    provenance.update(hashes)
    provenance["manifest_filename"] = manifest_path.name
    provenance["conflict_report_filename"] = conflict_path.name
    provenance["summary_filename"] = summary_path.name
    write_json(provenance, args.provenance)

    checksum_lines = [
        f"{provenance['raw_sha256']}  data/raw/{provenance['raw_filename']}",
        f"{hashes['manifest_sha256']}  data/manifests/{manifest_path.name}",
        f"{hashes['conflicts_sha256']}  data/manifests/{conflict_path.name}",
        f"{sha256_file(summary_path)}  data/manifests/{summary_path.name}",
    ]
    args.checksums.parent.mkdir(parents=True, exist_ok=True)
    args.checksums.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({**summary, **hashes}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
