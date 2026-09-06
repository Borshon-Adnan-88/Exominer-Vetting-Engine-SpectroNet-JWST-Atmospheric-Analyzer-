"""Resume per-host MAST discovery and freeze inventory only when complete."""
import argparse,json
from pathlib import Path
import pandas as pd
from kepler_scale.contracts import load_config,validate_inputs
from kepler_scale.discovery import discover_hosts,freeze_discovery
from kepler_scale.paths import resolve_data_paths
from kepler_lightcurves.provenance import sha256_file,write_json

def main():
 p=argparse.ArgumentParser(); p.add_argument("--data-root"); p.add_argument("--profile",choices=["laptop_safe","cloud_archive"],default="laptop_safe"); p.add_argument("--retry-failures",action="store_true"); p.add_argument("--discovery-concurrency",type=int); a=p.parse_args(); c=load_config(); validate_inputs(c); paths=resolve_data_paths(a.data_root,c,create=True); roster=pd.read_csv("data/stage4a/kepler_stage4a_host_roster_v1.csv")
 concurrency=a.discovery_concurrency or c["profiles"][a.profile]["discovery_concurrency"]
 if not 1<=concurrency<=4: raise ValueError("discovery concurrency must be between 1 and 4")
 def progress(n,total,r):
  if n%25==0 or n==total: print(f"discovery {n}/{total}: KIC {r['kepid']} {r['status']}",flush=True)
 records=discover_hosts(roster,paths.cache,c,concurrency,a.retry_failures,progress); inventory,outcomes=freeze_discovery(records)
 if len(outcomes)!=len(roster): raise RuntimeError("partial discovery cannot be frozen")
 out=Path("data/stage4a"); ip=out/"kepler_stage4a_mast_inventory_v1.csv"; op=out/"kepler_stage4a_discovery_outcomes_v1.csv"; inventory.to_csv(ip,index=False,lineterminator="\n"); outcomes.to_csv(op,index=False,lineterminator="\n")
 write_json({"hosts":len(outcomes),"products":len(inventory),"hosts_no_products":int((outcomes.status=="no_archive_product").sum()),"ambiguous_hosts":int((outcomes.status=="ambiguous_products").sum()),"retrieval_failure_hosts":int((outcomes.status=="retrieval_failure").sum()),"advertised_total_bytes":int(inventory.product_size.fillna(0).sum()),"inventory_sha256":sha256_file(ip),"outcomes_sha256":sha256_file(op),"operational_discovery_concurrency":concurrency},out/"kepler_stage4a_discovery_summary_v1.json")
 print(json.dumps(json.loads((out/"kepler_stage4a_discovery_summary_v1.json").read_text()),indent=2,sort_keys=True))
if __name__=="__main__": main()
