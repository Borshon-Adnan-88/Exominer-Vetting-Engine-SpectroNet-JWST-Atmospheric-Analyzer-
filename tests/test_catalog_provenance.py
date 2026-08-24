import hashlib

from kepler_catalog.provenance import sha256_file, write_json


def test_file_hash_and_sorted_json_are_deterministic(tmp_path):
    target = tmp_path / "value.bin"
    target.write_bytes(b"kepler-dr25")
    assert sha256_file(target) == hashlib.sha256(b"kepler-dr25").hexdigest()
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_json({"b": 2, "a": 1}, first)
    write_json({"a": 1, "b": 2}, second)
    assert first.read_bytes() == second.read_bytes()
