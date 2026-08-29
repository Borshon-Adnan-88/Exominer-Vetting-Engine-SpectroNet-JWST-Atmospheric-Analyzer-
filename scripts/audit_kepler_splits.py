"""Recompute Stage 3 acceptance audits from the frozen assignments."""

import json
import pandas as pd

from kepler_splits.audit import build_audit, enforce_acceptance
from kepler_splits.cohort import load_cohort


def main() -> None:
    config=json.loads(open("configs/kepler_split_policy_object_vs_host_v1.json",encoding="utf-8").read())
    cohort=load_cohort(config); assignments=pd.read_csv("data/splits/kepler_koi_split_assignments_v1.csv")
    _,summary=build_audit(assignments,cohort,config); print(json.dumps(summary,indent=2,sort_keys=True)); enforce_acceptance(summary)


if __name__ == "__main__": main()
