from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kepler_catalog.fields import SELECTED_FIELDS
from kepler_catalog.schema import CatalogValidationError, normalize_and_validate, read_catalog

FIXTURE = Path("tests/fixtures/kepler_dr25_koi_valid.csv")


def test_valid_fixture_has_frozen_schema_and_types():
    frame = read_catalog(FIXTURE)
    assert tuple(frame.columns) == SELECTED_FIELDS
    assert str(frame["kepid"].dtype) == "Int64"
    assert str(frame["koi_period"].dtype) == "Float64"
    assert frame["kepoi_name"].notna().all()


def test_missing_or_unexpected_columns_fail():
    frame = pd.read_csv(FIXTURE)
    with pytest.raises(CatalogValidationError, match="missing"):
        normalize_and_validate(frame.drop(columns="kepid"))
    frame["surprise"] = 1
    with pytest.raises(CatalogValidationError, match="unexpected"):
        normalize_and_validate(frame)


@pytest.mark.parametrize(("field", "value"), [("koi_score", 1.1), ("koi_period", np.inf), ("koi_fpflag_nt", 2)])
def test_invalid_numeric_values_fail(field, value):
    frame = pd.read_csv(FIXTURE)
    frame.loc[0, field] = value
    with pytest.raises(CatalogValidationError):
        normalize_and_validate(frame)


def test_invalid_fixture_fails():
    with pytest.raises(CatalogValidationError):
        read_catalog("tests/fixtures/kepler_dr25_koi_invalid.csv")
