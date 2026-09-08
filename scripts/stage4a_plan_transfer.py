"""Plan a bounded operational transfer without downloading any files."""

import argparse
import json
from pathlib import Path

import pandas as pd

from kepler_scale.contracts import load_config, validate_inputs
from kepler_scale.operations import get_operational_profile
from kepler_scale.paths import resolve_data_paths, resolve_scratch_paths
from kepler_scale.transfer import completed_verified_hosts, plan_host_transfer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root")
    parser.add_argument("--scratch-root")
    parser.add_argument("--profile", choices=["laptop_safe", "cloud_archive", "public_wifi", "home_wifi"], default="public_wifi")
    parser.add_argument("--max-download-bytes-per-run", type=int)
    args = parser.parse_args()
    config = load_config(); validate_inputs(config)
    profile = get_operational_profile(args.profile)
    ceiling = args.max_download_bytes_per_run if args.max_download_bytes_per_run is not None else profile["max_download_bytes_per_run"]
    data_paths = resolve_data_paths(args.data_root, config, create=True)
    scratch_paths = resolve_scratch_paths(args.scratch_root, data_paths, create=True)
    roster = pd.read_csv("data/stage4a/kepler_stage4a_host_roster_v1.csv")
    inventory = pd.read_csv("data/stage4a/kepler_stage4a_mast_inventory_v1.csv")
    state_dir = data_paths.cache / "host_state"
    states = []
    for path in sorted(state_dir.glob("*.json")) if state_dir.exists() else []:
        try: states.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError): continue
    plan = plan_host_transfer(roster, inventory, completed_verified_hosts(states), ceiling)
    # Absolute roots are intentionally displayed only on stdout and never serialized.
    print(json.dumps({
        "profile": args.profile,
        "data_root_runtime": str(data_paths.root),
        "scratch_root_runtime": str(scratch_paths.root),
        "planned_hosts": len(plan.kepids),
        "planned_bytes": plan.planned_bytes,
        "ceiling_bytes": plan.ceiling_bytes,
        "stop_reason": plan.stop_reason,
        "next_incomplete_kepid": plan.next_incomplete_kepid,
        "copy_raw_to_data_root": profile["copy_raw_to_data_root"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__": main()
