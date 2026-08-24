import pytest

import kepler_catalog.download as download
from kepler_catalog.download import TapSchemaError


def test_missing_live_field_aborts_before_catalogue_request(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "get_live_columns", lambda timeout=120: ({"kepid"}, b"metadata"))
    catalogue_requested = False

    def unexpected_request(query, timeout=120):
        nonlocal catalogue_requested
        catalogue_requested = True
        return b""

    monkeypatch.setattr(download, "_tap_request", unexpected_request)
    with pytest.raises(TapSchemaError, match="retrieval aborted"):
        download.retrieve_snapshot("SELECT * FROM q1_q17_dr25_koi", tmp_path)
    assert catalogue_requested is False
