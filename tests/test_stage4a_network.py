import os
import pytest
from kepler_lightcurves.mast import discover_targets

@pytest.mark.network
@pytest.mark.skipif(os.getenv("RUN_NETWORK_TESTS")!="1",reason="set RUN_NETWORK_TESTS=1 to enable")
def test_stage4a_batch_discovery_returns_each_requested_host_once():
    result=discover_targets([11904151,9636135])
    assert set(result)=={11904151,9636135}
    for kepid,products in result.items():
        assert not products.empty and products.kepid.eq(kepid).all()
        assert products.quarter.between(1,17).all()
        assert not products.duplicated(["kepid","quarter"]).any()
