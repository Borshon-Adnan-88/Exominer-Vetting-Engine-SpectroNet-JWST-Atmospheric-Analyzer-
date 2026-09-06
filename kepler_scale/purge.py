"""Strict process-and-purge eligibility validation."""
from pathlib import Path

REQUIRED={"inventory_frozen","sizes_verified","fits_sha256_recorded","fits_identity_validated","all_kois_final","constructed_arrays_written","array_checksums_verified","provenance_written"}

def assert_purge_eligible(state:dict,host_raw_dir:str|Path,raw_root:str|Path,allow_purge:bool)->list[Path]:
    if not allow_purge: raise PermissionError("explicit --purge-verified-raw authorization required")
    missing=sorted(k for k in REQUIRED if state.get(k) is not True)
    if missing: raise RuntimeError(f"host is not purge eligible: {missing}")
    host=Path(host_raw_dir).resolve(); root=Path(raw_root).resolve()
    try: host.relative_to(root)
    except ValueError as exc: raise RuntimeError("purge path escapes raw FITS root") from exc
    if host==root: raise RuntimeError("refusing to purge raw FITS root")
    return sorted((p for p in host.iterdir() if p.is_file() and p.name.endswith("_llc.fits")),key=lambda p:p.name)
