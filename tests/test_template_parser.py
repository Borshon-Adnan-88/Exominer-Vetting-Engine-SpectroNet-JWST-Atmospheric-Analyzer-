from pathlib import Path

import pytest

from spectronet.preprocessing import GAS_DEFAULTS, parse_planet_template


@pytest.mark.parametrize("filename", ["wasp39b.txt", "habitable_target.txt"])
def test_supplied_templates_parse(filename):
    content = (Path("sample_planet_upload") / filename).read_bytes()
    parsed = parse_planet_template(content)
    assert set(parsed) == set(GAS_DEFAULTS)
    assert all(0.0 <= value <= 1.0 for value in parsed.values())


def test_comments_whitespace_case_and_first_equals_split():
    parsed = parse_planet_template(b" ch4 = 0.2 # comment\nCO2 = 0.3\n")
    assert parsed["CH4"] == pytest.approx(0.2)
    assert parsed["CO2"] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="numeric"):
        parse_planet_template(b"CH4 = 0.2=unexpected")


@pytest.mark.parametrize(
    "content",
    [
        b"CH4 = nan",
        b"CH4 = inf",
        b"CH4 = -0.1",
        b"CH4 = 1.1",
        b"CH4 = nope",
        b"UNKNOWN = 0.2",
        b"CH4 0.2",
        b"\xff",
    ],
)
def test_invalid_templates_raise_useful_errors(content):
    with pytest.raises(ValueError):
        parse_planet_template(content)


def test_non_bytes_input_is_rejected():
    with pytest.raises(TypeError):
        parse_planet_template("CH4 = 0.2")
