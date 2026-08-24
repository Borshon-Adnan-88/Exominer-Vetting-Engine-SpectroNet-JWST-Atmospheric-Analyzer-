from kepler_catalog.manifest import build_manifest


def test_kepid_is_complete_and_multi_koi_hosts_are_counted():
    manifest, _, summary = build_manifest("tests/fixtures/kepler_dr25_koi_valid.csv")
    assert manifest["kepid"].notna().all()
    assert manifest.groupby("kepid").ngroups == 5
    assert summary["multi_koi_hosts"] == 1
    assert len(manifest.loc[manifest["kepid"].eq(1001)]) == 2
