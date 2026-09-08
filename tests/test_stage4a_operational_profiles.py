from kepler_scale.contracts import load_config
from types import SimpleNamespace
from pathlib import Path
from kepler_scale.operations import get_operational_profile, raw_download_root


def test_public_wifi_defaults_and_existing_profiles_unchanged():
    scientific = load_config()["profiles"]
    for name in ("laptop_safe", "cloud_archive"):
        operational = get_operational_profile(name)
        for key in ("host_batch_size", "cpu_workers", "download_concurrency", "retain_raw_fits"):
            assert operational[key] == scientific[name][key]
    public = get_operational_profile("public_wifi")
    assert public == {
        "host_batch_size": 1, "cpu_workers": 1, "download_concurrency": 1,
        "retain_raw_fits": False, "copy_raw_to_data_root": False,
        "process_and_purge": True, "max_download_bytes_per_run": 100 * 1024 * 1024,
    }


def test_home_wifi_defaults_use_scratch_without_raw_archive():
    home = get_operational_profile("home_wifi")
    assert home == {
        "host_batch_size": 10, "cpu_workers": 1, "download_concurrency": 1,
        "retain_raw_fits": False, "copy_raw_to_data_root": False,
        "process_and_purge": True, "max_download_bytes_per_run": None,
    }


def test_operational_values_are_absent_from_scientific_configuration():
    config = load_config()
    assert "public_wifi" not in config["profiles"]
    assert "scratch" not in str(config).lower()
    assert "max_download_bytes_per_run" not in str(config)


def test_public_wifi_raw_download_has_no_automatic_data_root_copy():
    data=SimpleNamespace(raw_fits=Path("data-root/raw_fits")); scratch=SimpleNamespace(raw_fits=Path("scratch/raw_fits"))
    assert raw_download_root(get_operational_profile("public_wifi"),data,scratch)==scratch.raw_fits
    assert raw_download_root(get_operational_profile("home_wifi"),data,scratch)==scratch.raw_fits
    assert raw_download_root(get_operational_profile("cloud_archive"),data,scratch)==data.raw_fits
