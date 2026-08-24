from pathlib import Path

import pandas as pd

from kepler_catalog.labels import apply_label_policy
from kepler_catalog.schema import read_catalog

FIXTURE = Path("tests/fixtures/kepler_dr25_koi_valid.csv")


def test_versioned_policy_and_conflicts_match_golden_fixture():
    result = apply_label_policy(read_catalog(FIXTURE))
    columns = ["kepid", "kepoi_name", "label", "label_name", "inclusion_status", "exclusion_reason", "label_conflict"]
    expected = pd.read_csv("tests/fixtures/expected_kepler_dr25_manifest.csv")
    expected["label"] = expected["label"].astype("Int64")
    normalize = lambda frame: frame.map(lambda value: "<missing>" if pd.isna(value) else str(value))
    pd.testing.assert_frame_equal(
        normalize(result[columns].reset_index(drop=True)),
        normalize(expected),
    )
    assert result["label_policy"].eq("confirmed_vs_false_positive_v1").all()


def test_conflict_audit_reasons_are_preserved():
    result = apply_label_policy(read_catalog(FIXTURE))
    actual = result.loc[result["label_conflict"], ["kepid", "kepoi_name", "kepler_name", "koi_disposition", "koi_pdisposition", "conflict_reason"]]
    expected = pd.read_csv("tests/fixtures/kepler_dr25_koi_conflicts.csv").rename(columns={"expected_conflict_reason": "conflict_reason"})
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected, check_dtype=False)
