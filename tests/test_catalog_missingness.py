import pandas as pd
import pytest

from kepler_catalog.labels import apply_label_policy
from kepler_catalog.schema import CatalogValidationError, normalize_and_validate


@pytest.mark.parametrize("field", ["kepid", "kepoi_name", "koi_disposition"])
def test_missing_identity_or_disposition_fails(field):
    frame = pd.read_csv("tests/fixtures/kepler_dr25_koi_valid.csv")
    frame.loc[0, field] = None
    with pytest.raises(CatalogValidationError, match=field):
        normalize_and_validate(frame)


@pytest.mark.parametrize("field", ["koi_period", "koi_time0bk", "koi_duration"])
def test_missing_required_ephemeris_on_included_row_fails(field):
    frame = pd.read_csv("tests/fixtures/kepler_dr25_koi_valid.csv")
    frame.loc[0, field] = None
    with pytest.raises(CatalogValidationError, match=field):
        apply_label_policy(normalize_and_validate(frame))


def test_optional_stellar_missingness_is_preserved():
    frame = normalize_and_validate(pd.read_csv("tests/fixtures/kepler_dr25_koi_valid.csv"))
    assert pd.isna(frame.loc[3, "koi_smet"])
