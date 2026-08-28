"""Select and freeze the deterministic 16-KOI Stage 2 pilot."""

import json
from pathlib import Path

from kepler_lightcurves.pilot import select_pilot
from kepler_lightcurves.provenance import write_json


def main() -> None:
    config = Path("configs/kepler_pilot_cohort_v1.json")
    ranking, selected, summary = select_pilot("data/manifests/kepler_q1_q17_dr25_koi_manifest.csv", config)
    output = Path("data/pilot"); output.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output / "kepler_pilot_eligible_ranking.csv", index=False, lineterminator="\n", float_format="%.15g")
    selected.to_csv(output / "kepler_pilot_selection.csv", index=False, lineterminator="\n", float_format="%.15g")
    summary["cell_counts"] = selected.groupby(["label", "host_category", "period_stratum"]).size().rename("count").reset_index().to_dict("records")
    write_json(summary, output / "kepler_pilot_summary.json")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(selected[["pilot_order", "kepoi_name", "kepid", "label", "host_category", "period_stratum", "koi_period", "koi_duration", "koi_model_snr"]].to_string(index=False))


if __name__ == "__main__":
    main()
