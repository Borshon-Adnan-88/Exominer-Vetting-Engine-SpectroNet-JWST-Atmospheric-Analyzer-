"""Operational profiles that cannot change scientific outputs."""

import json
from pathlib import Path


def load_operational_profiles(path: str | Path = "configs/kepler_stage4a_operational_profiles_v1.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_operational_profile(name: str, path: str | Path = "configs/kepler_stage4a_operational_profiles_v1.json") -> dict:
    profiles = load_operational_profiles(path)["profiles"]
    if name not in profiles:
        raise ValueError(f"unknown operational profile: {name}")
    return dict(profiles[name])


def raw_download_root(profile: dict, data_paths, scratch_paths) -> Path:
    """Choose one download destination; public Wi-Fi never stages a cloud copy."""
    return data_paths.raw_fits if profile["retain_raw_fits"] else scratch_paths.raw_fits
