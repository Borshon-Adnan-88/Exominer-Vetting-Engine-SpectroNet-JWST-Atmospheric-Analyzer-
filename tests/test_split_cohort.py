import json
import pandas as pd
import pytest

from kepler_splits.cohort import load_cohort
from kepler_splits.schemas import SplitValidationError


def config(): return json.loads(open("configs/kepler_split_policy_object_vs_host_v1.json",encoding="utf-8").read())


def test_frozen_cohort_membership_and_order():
    cohort=load_cohort(config())
    assert len(cohort)==6637 and cohort.kepid.nunique()==5804
    assert cohort.label.value_counts().to_dict()=={0:3963,1:2674}
    assert cohort.equals(cohort.sort_values(["kepid","kepoi_name","source_row_index"],kind="stable").reset_index(drop=True))


def test_source_hash_is_enforced(tmp_path):
    c=config(); p=tmp_path/"changed.csv"; pd.DataFrame({"x":[1]}).to_csv(p,index=False); c["source_manifest"]=str(p)
    with pytest.raises(SplitValidationError,match="SHA-256"): load_cohort(c)
