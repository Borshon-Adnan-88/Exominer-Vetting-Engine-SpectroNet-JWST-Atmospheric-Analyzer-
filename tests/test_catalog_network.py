import os

import pytest

from kepler_catalog.download import validate_live_schema
from kepler_catalog.fields import SELECTED_FIELDS


@pytest.mark.network
@pytest.mark.skipif(os.getenv("RUN_NETWORK_TESTS") != "1", reason="set RUN_NETWORK_TESTS=1 to enable")
def test_live_tap_schema_contains_every_requested_field():
    columns, metadata_bytes = validate_live_schema()
    assert set(SELECTED_FIELDS) <= columns
    assert metadata_bytes
