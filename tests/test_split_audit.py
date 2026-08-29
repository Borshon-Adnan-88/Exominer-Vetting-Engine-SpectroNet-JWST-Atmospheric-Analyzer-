from kepler_splits.audit import build_audit


def test_acceptance_audits_and_metadata_missingness(split_config, split_cohort, split_assignments):
    audit,summary=build_audit(split_assignments,split_cohort,split_config)
    assert summary["acceptance_pass"] and not summary["tolerance_failures"]
    assert len(audit)==60 and audit.koi_model_snr_missing.sum()==55*20
    grouped=[r for r in summary["strategy_repeats"] if r["strategy"]=="host_group_stratified"]
    assert all(r["grouped_host_overlap"]==0 and r["mixed_label_hosts_intact"]==36 for r in grouped)
