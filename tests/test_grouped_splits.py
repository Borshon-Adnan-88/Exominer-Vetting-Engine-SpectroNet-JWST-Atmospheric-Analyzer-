def test_grouped_assignments_keep_every_host_and_mixed_host_intact(split_assignments, split_cohort):
    g=split_assignments[split_assignments.strategy=="host_group_stratified"]
    assert len(g)==66370
    assert g.groupby(["repeat_id","kepid"]).partition.nunique().eq(1).all()
    mixed=set(split_cohort.groupby("kepid").label.nunique().loc[lambda x:x>1].index)
    assert len(mixed)==36 and g[g.kepid.isin(mixed)].groupby(["repeat_id","kepid"]).partition.nunique().eq(1).all()
