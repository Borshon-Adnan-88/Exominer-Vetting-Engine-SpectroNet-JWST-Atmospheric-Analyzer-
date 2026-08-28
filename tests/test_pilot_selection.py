from kepler_lightcurves.pilot import select_pilot


def test_real_frozen_manifest_yields_complete_deterministic_pilot():
    first = select_pilot("data/manifests/kepler_q1_q17_dr25_koi_manifest.csv", "configs/kepler_pilot_cohort_v1.json")
    second = select_pilot("data/manifests/kepler_q1_q17_dr25_koi_manifest.csv", "configs/kepler_pilot_cohort_v1.json")
    _, selected, summary = first
    assert len(selected) == summary["selected_count"] == 16
    assert selected["kepid"].nunique() == 16
    assert selected.groupby(["label", "host_category", "period_stratum"]).size().eq(1).all()
    assert selected["koi_time0bk"].notna().all()
    assert first[1]["kepoi_name"].tolist() == second[1]["kepoi_name"].tolist()
