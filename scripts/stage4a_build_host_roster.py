"""Build deterministic label-blind host roster and profile batches."""
import argparse,json
from pathlib import Path
from kepler_scale.batches import build_roster
from kepler_scale.contracts import load_cohort,load_config
from kepler_lightcurves.provenance import sha256_file,write_json

def main():
 p=argparse.ArgumentParser(); p.add_argument("--profile",choices=["laptop_safe","cloud_archive"],default="laptop_safe"); a=p.parse_args(); c=load_config(); cohort=load_cohort(c); roster,batches=build_roster(cohort,c["profiles"][a.profile]["host_batch_size"]); out=Path("data/stage4a"); out.mkdir(exist_ok=True)
 rp=out/"kepler_stage4a_host_roster_v1.csv"; bp=out/"kepler_stage4a_batches_v1.csv"; roster.to_csv(rp,index=False,lineterminator="\n"); batches.to_csv(bp,index=False,lineterminator="\n")
 write_json({"profile":a.profile,"hosts":len(roster),"kois":int(roster.koi_count.sum()),"batches":len(batches),"roster_sha256":sha256_file(rp),"batches_sha256":sha256_file(bp)},out/"kepler_stage4a_roster_summary_v1.json")
 print(json.dumps({"hosts":len(roster),"batches":len(batches)},indent=2))
if __name__=="__main__": main()
