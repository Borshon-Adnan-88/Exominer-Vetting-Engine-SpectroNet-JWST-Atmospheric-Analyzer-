"""Read-only preflight after complete frozen discovery."""
import argparse,json
from pathlib import Path
import pandas as pd
from kepler_scale.contracts import load_config,validate_inputs
from kepler_scale.paths import resolve_data_paths
from kepler_scale.preflight import build_preflight
from kepler_lightcurves.provenance import sha256_file,write_json

def main():
 p=argparse.ArgumentParser(); p.add_argument("--data-root"); p.add_argument("--profile",choices=["laptop_safe","cloud_archive"],default="laptop_safe"); a=p.parse_args(); c=load_config(); inputs=validate_inputs(c); paths=resolve_data_paths(a.data_root,c,create=False); root=Path("data/stage4a"); roster=pd.read_csv(root/"kepler_stage4a_host_roster_v1.csv"); batches=pd.read_csv(root/"kepler_stage4a_batches_v1.csv"); inventory=pd.read_csv(root/"kepler_stage4a_mast_inventory_v1.csv"); outcomes=pd.read_csv(root/"kepler_stage4a_discovery_outcomes_v1.csv")
 if len(outcomes)!=inputs["hosts"]: raise RuntimeError("discovery is incomplete")
 report=build_preflight(c,a.profile,paths,roster,batches,inventory,outcomes); report.update({"inventory_sha256":sha256_file(root/"kepler_stage4a_mast_inventory_v1.csv"),"stage4a_config_sha256":sha256_file("configs/kepler_stage4a_scale_v1.json")}); write_json(report,root/f"kepler_stage4a_preflight_{a.profile}_v1.json"); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
