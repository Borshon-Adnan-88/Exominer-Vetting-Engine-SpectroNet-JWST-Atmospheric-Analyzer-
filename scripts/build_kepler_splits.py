"""Build and audit all fixed Stage 3 split assignments."""

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from kepler_splits.assign import build_assignments
from kepler_splits.audit import build_audit, enforce_acceptance
from kepler_splits.cohort import load_cohort
from kepler_splits.provenance import sha256_file, write_json

CONFIG = Path("configs/kepler_split_policy_object_vs_host_v1.json")
OUTPUT = Path("data/splits")


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8")); cohort = load_cohort(config)
    assignments = build_assignments(cohort, config); audit, summary = build_audit(assignments, cohort, config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assignment_path = OUTPUT / "kepler_koi_split_assignments_v1.csv"
    audit_path = OUTPUT / "kepler_koi_split_audit_v1.csv"
    summary_path = OUTPUT / "kepler_koi_split_summary_v1.json"
    assignments.to_csv(assignment_path, index=False, lineterminator="\n")
    audit.to_csv(audit_path, index=False, lineterminator="\n", float_format="%.15g")
    summary.update({"split_policy_version":config["split_policy_version"],"label_policy":config["label_policy"],"source_manifest_sha256":config["source_manifest_sha256"],"config_sha256":sha256_file(CONFIG),"assignment_sha256":sha256_file(assignment_path),"audit_sha256":sha256_file(audit_path)})
    write_json(summary, summary_path)
    checksums = {"config": sha256_file(CONFIG), "assignments": sha256_file(assignment_path), "audit": sha256_file(audit_path), "summary": sha256_file(summary_path)}
    (OUTPUT / "checksums.sha256").write_text("".join(f"{digest}  {name}\n" for name,digest in [("configs/kepler_split_policy_object_vs_host_v1.json",checksums["config"]),("data/splits/kepler_koi_split_assignments_v1.csv",checksums["assignments"]),("data/splits/kepler_koi_split_audit_v1.csv",checksums["audit"]),("data/splits/kepler_koi_split_summary_v1.json",checksums["summary"]) ]),encoding="utf-8")
    try: revision=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError): revision=None
    dirty=bool(subprocess.run(["git","status","--porcelain"],capture_output=True,text=True,check=False).stdout.strip())
    write_json({"built_at_utc":datetime.now(timezone.utc).isoformat(),"git_revision":revision,"worktree_dirty":dirty,"environment":{"python":sys.version.split()[0],"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__},**checksums},OUTPUT/"kepler_koi_split_provenance_v1.json")
    print(json.dumps({**checksums,"acceptance_pass":summary["acceptance_pass"],"assignment_rows":len(assignments)},indent=2,sort_keys=True))
    enforce_acceptance(summary)


if __name__ == "__main__": main()
