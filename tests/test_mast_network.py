import os

import pytest

from kepler_lightcurves.mast import discover_target


@pytest.mark.network
@pytest.mark.skipif(os.getenv("RUN_NETWORK_TESTS") != "1", reason="set RUN_NETWORK_TESTS=1 to enable")
def test_mast_discovers_official_kepler_llc_products():
    result = discover_target(11904151)
    assert not result.empty
    assert result["product_filename"].str.lower().str.endswith("_llc.fits").all()
    assert result["quarter"].between(1, 17).all()
    assert not result.duplicated(["kepid", "quarter"]).any()
