"""Run one explicit frozen Stage 4A host batch."""
import argparse,json
from pathlib import Path
import pandas as pd
from kepler_scale.atomic import atomic_json
from kepler_scale.contracts import load_cohort,load_config,validate_inputs
from kepler_scale.operations import get_operational_profile
from kepler_scale.paths import resolve_data_paths,resolve_scratch_paths
from kepler_scale.runner import run_batch

def main():
 p=argparse.ArgumentParser(); p.add_argument("--batch-id",required=True); p.add_argument("--profile",choices=["home_wifi"],required=True); p.add_argument("--data-root"); p.add_argument("--scratch-root"); p.add_argument("--no-purge",action="store_true"); a=p.parse_args(); config=load_config(); validate_inputs(config); profile=get_operational_profile(a.profile)
 data=resolve_data_paths(a.data_root,config,create=True); scratch=resolve_scratch_paths(a.scratch_root,data,create=True); roster=pd.read_csv("data/stage4a/kepler_stage4a_host_roster_v1.csv"); inventory=pd.read_csv("data/stage4a/kepler_stage4a_mast_inventory_v1.csv"); cohort=load_cohort(config); scientific=json.loads(Path(config["expected_inputs"]["stage2_config"]).read_text(encoding="utf-8")); scientific["retry"]=config["retry"]
 report=run_batch(a.batch_id,roster,inventory,cohort,scientific,profile,data,scratch,purge=not a.no_purge,rebuild_first_host=True); atomic_json(data.cache/"benchmarks"/f"{a.batch_id}-{a.profile}.json",report); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=="__main__": main()
