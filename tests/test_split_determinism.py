import numpy as np
from kepler_splits.assign import _folds


def test_determinism_and_fixed_seeds_differ(split_config, split_cohort, split_assignments):
    for strategy in split_config["strategies"]:
        assert np.array_equal(_folds(split_cohort,strategy,1729,split_config),_folds(split_cohort,strategy,1729,split_config))
    for strategy,block in split_assignments.groupby("strategy"):
        pivot=block.pivot(index="kepoi_name",columns="repeat_id",values="partition")
        assert any(not pivot[0].equals(pivot[x]) for x in range(1,10))
