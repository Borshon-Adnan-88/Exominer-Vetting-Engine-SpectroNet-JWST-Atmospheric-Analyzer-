import json

import pandas as pd

from kepler_catalog.manifest import build_manifest, write_manifest_artifacts


def test_summary_matches_golden_fixture():
    _, _, summary = build_manifest("tests/fixtures/kepler_dr25_koi_valid.csv")
    expected = json.loads(open("tests/fixtures/expected_kepler_dr25_summary.json", encoding="utf-8").read())
    assert summary == expected


def test_output_is_byte_reproducible_and_input_order_independent(tmp_path):
    source = pd.read_csv("tests/fixtures/kepler_dr25_koi_valid.csv")
    shuffled = tmp_path / "shuffled.csv"
    source.sample(frac=1, random_state=9).to_csv(shuffled, index=False)
    first = build_manifest("tests/fixtures/kepler_dr25_koi_valid.csv")
    second = build_manifest(shuffled)
    paths = []
    for index, artifacts in enumerate((first, second)):
        directory = tmp_path / str(index)
        manifest_path = directory / "manifest.csv"
        write_manifest_artifacts(*artifacts, manifest_path, directory / "conflicts.csv", directory / "summary.json")
        paths.append(manifest_path)
    assert paths[0].read_bytes() == paths[1].read_bytes()
