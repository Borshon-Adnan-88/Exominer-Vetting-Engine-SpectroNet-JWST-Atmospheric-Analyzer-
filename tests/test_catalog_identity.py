import pandas as pd
import pytest

from kepler_catalog.schema import CatalogValidationError, normalize_and_validate


def test_duplicate_kepoi_name_fails():
    frame = pd.read_csv("tests/fixtures/kepler_dr25_koi_valid.csv")
    frame.loc[1, "kepoi_name"] = frame.loc[0, "kepoi_name"]
    with pytest.raises(CatalogValidationError, match="kepoi_name must be unique"):
        normalize_and_validate(frame)


def test_duplicate_complete_federation_key_fails():
    frame = pd.read_csv("tests/fixtures/kepler_dr25_koi_valid.csv")
    frame.loc[1, "koi_tce_plnt_num"] = frame.loc[0, "koi_tce_plnt_num"]
    with pytest.raises(CatalogValidationError, match="federated TCE key"):
        normalize_and_validate(frame)
