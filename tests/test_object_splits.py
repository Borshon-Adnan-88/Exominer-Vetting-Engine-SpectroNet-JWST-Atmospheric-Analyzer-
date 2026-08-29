def test_object_assignments_are_complete_and_can_fragment_hosts(split_assignments):
    o=split_assignments[split_assignments.strategy=="object_stratified"]
    assert len(o)==66370
    assert o.groupby(["repeat_id","kepoi_name"]).size().eq(1).all()
    assert (o.groupby(["repeat_id","kepid"]).partition.nunique()>1).any()
