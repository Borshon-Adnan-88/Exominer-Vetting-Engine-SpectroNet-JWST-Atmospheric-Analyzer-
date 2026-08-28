"""Independent per-KOI construction of global and local transit views."""

from pathlib import Path

import numpy as np

from .binning import median_bin
from .fold import local_coordinate, phase_fold, validate_ephemeris
from .normalize import normalize_quarter
from .quality import quality_keep_mask
from .schemas import LightCurveError, QuarterData


def _qc_reason(condition: bool, name: str, output: list[str]) -> None:
    if condition:
        output.append(name)


def process_object(quarters: list[QuarterData], period: float, epoch: float, duration: float, config: dict) -> tuple[dict[str, np.ndarray], dict]:
    validate_ephemeris(period, epoch, duration)
    if not quarters:
        raise LightCurveError("no_usable_photometry")
    times, fluxes, errors, saps, quarter_ids = [], [], [], [], []
    raw_times, quarter_stats = [], []
    for quarter in sorted(quarters, key=lambda item: item.quarter):
        raw_finite = quarter.time[np.isfinite(quarter.time)]
        raw_times.extend(raw_finite.tolist())
        keep_quality = quality_keep_mask(quarter.quality, config["quality_mask"]["integer"])
        finite = np.isfinite(quarter.time) & np.isfinite(quarter.pdcsap_flux) & np.isfinite(quarter.pdcsap_flux_err) & (quarter.pdcsap_flux_err > 0)
        keep = keep_quality & finite
        if not keep.any():
            quarter_stats.append({"quarter": quarter.quarter, "usable": 0, "total": len(keep)})
            continue
        normalized, normalized_err, median = normalize_quarter(quarter.pdcsap_flux[keep], quarter.pdcsap_flux_err[keep])
        times.append(quarter.time[keep]); fluxes.append(normalized); errors.append(normalized_err)
        sap = quarter.sap_flux[keep]
        sap_median = np.nanmedian(sap)
        saps.append(sap / sap_median if np.isfinite(sap_median) and sap_median > 0 else np.full_like(sap, np.nan))
        quarter_ids.append(np.full(keep.sum(), quarter.quarter, dtype=np.int16))
        quarter_stats.append({"quarter": quarter.quarter, "usable": int(keep.sum()), "total": int(len(keep)), "median": median, "cadence_seconds": quarter.actual_cadence_seconds})
    if not times:
        raise LightCurveError("no_usable_photometry")
    time = np.concatenate(times); flux = np.concatenate(fluxes); flux_err = np.concatenate(errors)
    sap_flux = np.concatenate(saps); quarter_id = np.concatenate(quarter_ids)
    order = np.argsort(time, kind="stable")
    time, flux, flux_err, sap_flux, quarter_id = time[order], flux[order], flux_err[order], sap_flux[order], quarter_id[order]
    phase = phase_fold(time, period, epoch)
    local = local_coordinate(phase, period, duration)
    global_cfg, local_cfg = config["global_view"], config["local_view"]
    global_view = median_bin(phase, flux, global_cfg["minimum"], global_cfg["maximum"], global_cfg["bins"], circular=True)
    local_view = median_bin(local, flux, local_cfg["minimum"], local_cfg["maximum"], local_cfg["bins"], circular=False)
    local_sap_view = median_bin(local, sap_flux, local_cfg["minimum"], local_cfg["maximum"], local_cfg["bins"], circular=False)
    raw_span = max(raw_times) - min(raw_times) if len(raw_times) > 1 else 0.0
    usable_span = float(time[-1] - time[0]) if len(time) > 1 else 0.0
    baseline_fraction = usable_span / raw_span if raw_span > 0 else 0.0
    transit_index = np.rint((time - epoch) / period).astype(np.int64)
    in_transit = np.abs(local) <= 0.5
    observed_epochs = int(np.unique(transit_index[in_transit]).size)
    qc = config["pilot_qc_flags"]
    reasons: list[str] = []
    _qc_reason(len(set(quarter_id.tolist())) < qc["minimum_valid_quarters"], "valid_quarters_below_8", reasons)
    _qc_reason(baseline_fraction < qc["minimum_temporal_baseline_fraction"], "temporal_baseline_fraction_below_0.50", reasons)
    _qc_reason(len(time) < qc["minimum_total_valid_cadences"], "valid_cadences_below_1000", reasons)
    _qc_reason(observed_epochs < qc["minimum_distinct_observed_transit_epochs"], "observed_transit_epochs_below_3", reasons)
    _qc_reason(int(in_transit.sum()) < qc["minimum_valid_in_transit_cadences"], "in_transit_cadences_below_10", reasons)
    _qc_reason(global_view.observed_fraction < qc["minimum_global_observed_bin_fraction"], "global_coverage_below_0.80", reasons)
    _qc_reason(local_view.observed_fraction < qc["minimum_local_observed_bin_fraction"], "local_coverage_below_0.10", reasons)
    _qc_reason(global_view.longest_missing_run_fraction > qc["maximum_global_missing_run_fraction"], "global_missing_run_above_0.10", reasons)
    _qc_reason(local_view.longest_missing_run_fraction > qc["maximum_local_missing_run_fraction"], "local_missing_run_above_0.25", reasons)
    arrays = {
        "global_flux": global_view.flux, "global_observed_mask": global_view.observed_mask,
        "global_count": global_view.count, "local_flux": local_view.flux,
        "local_observed_mask": local_view.observed_mask, "local_count": local_view.count,
    }
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key.endswith("flux")):
        raise LightCurveError("nonfinite_constructed_output")
    center_slice = slice(80, 120)
    outer = np.r_[local_view.flux[:40], local_view.flux[160:]]
    pdcsap_depth = float(np.median(outer) - np.median(local_view.flux[center_slice]))
    sap_outer = np.r_[local_sap_view.flux[:40], local_sap_view.flux[160:]]
    sap_depth = float(np.median(sap_outer) - np.median(local_sap_view.flux[center_slice]))
    depth_scale = max(abs(pdcsap_depth), 1e-6)
    morphology_difference = abs(sap_depth - pdcsap_depth) / depth_scale
    local_width = (local_cfg["maximum"] - local_cfg["minimum"]) / local_cfg["bins"]
    local_centers = np.linspace(local_cfg["minimum"], local_cfg["maximum"], local_cfg["bins"], endpoint=False) + local_width / 2
    minimum_coordinate = float(local_centers[int(np.argmin(local_view.flux))])
    edge_medians = []
    for q in sorted(set(quarter_id.tolist())):
        qflux = flux[quarter_id == q]
        edge_medians.append((q, float(np.median(qflux[:min(100, len(qflux))])), float(np.median(qflux[-min(100, len(qflux)):]))))
    boundary_jumps = [abs(edge_medians[i][1] - edge_medians[i-1][2]) for i in range(1, len(edge_medians))]
    metadata = {
        "processing_status": "constructed", "qc_pass": not reasons, "qc_reasons": reasons,
        "valid_quarters": sorted(set(quarter_id.tolist())), "quarter_stats": quarter_stats,
        "valid_cadences": int(len(time)), "raw_temporal_span_days": raw_span,
        "usable_temporal_span_days": usable_span, "temporal_baseline_fraction": baseline_fraction,
        "observed_transit_epochs": observed_epochs, "in_transit_cadences": int(in_transit.sum()),
        "global_observed_fraction": global_view.observed_fraction,
        "local_observed_fraction": local_view.observed_fraction,
        "global_missing_run_fraction": global_view.longest_missing_run_fraction,
        "local_missing_run_fraction": local_view.longest_missing_run_fraction,
        "phase_center_flux": float(global_view.flux[len(global_view.flux)//2]),
        "local_minimum_coordinate_durations": minimum_coordinate,
        "visibly_miscentered_flag": abs(minimum_coordinate) > 0.5,
        "phase_inversion_flag": pdcsap_depth <= 0,
        "pdcsap_local_depth": pdcsap_depth, "sap_local_depth": sap_depth,
        "sap_pdcsap_morphology_difference_fraction": morphology_difference,
        "sap_pdcsap_material_difference_flag": morphology_difference > 0.20,
        "maximum_quarter_boundary_jump": max(boundary_jumps, default=0.0),
        "strong_quarter_discontinuity_flag": max(boundary_jumps, default=0.0) > 0.01,
        "unusually_high_interpolation_flag": (
            1 - global_view.observed_fraction > config["diagnostic_review_flags"]["maximum_interpolated_fraction"]
            or 1 - local_view.observed_fraction > config["diagnostic_review_flags"]["maximum_interpolated_fraction"]
        ),
    }
    return arrays, metadata


def save_arrays(path: str | Path, arrays: dict[str, np.ndarray]) -> str:
    import hashlib
    import io
    import zipfile
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for key in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(arrays[key]), allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return hashlib.sha256(path.read_bytes()).hexdigest()
