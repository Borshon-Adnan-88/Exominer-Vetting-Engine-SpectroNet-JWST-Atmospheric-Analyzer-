"""Freeze pre-download metadata checksums and separate timestamped provenance."""
import json,platform,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
import astropy,astroquery,numpy as np,pandas as pd
from kepler_lightcurves.provenance import sha256_file,write_json

def main():
 root=Path("data/stage4a"); files=[Path("configs/kepler_stage4a_scale_v1.json"),root/"kepler_stage4a_host_roster_v1.csv",root/"kepler_stage4a_batches_v1.csv",root/"kepler_stage4a_roster_summary_v1.json",root/"kepler_stage4a_mast_inventory_v1.csv",root/"kepler_stage4a_discovery_outcomes_v1.csv",root/"kepler_stage4a_discovery_summary_v1.json",root/"kepler_stage4a_preflight_laptop_safe_v1.json",root/"kepler_stage4a_preflight_cloud_archive_v1.json"]
 checks=[(sha256_file(path),path.as_posix()) for path in files]; checksum_path=root/"checksums.sha256"; checksum_path.write_text("".join(f"{digest}  {name}\n" for digest,name in checks),encoding="utf-8")
 revision=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip(); dirty=bool(subprocess.run(["git","status","--porcelain"],capture_output=True,text=True).stdout.strip())
 config=json.loads(Path("configs/kepler_stage4a_scale_v1.json").read_text())
 provenance={"pipeline_id":config["pipeline_id"],"built_at_utc":datetime.now(timezone.utc).isoformat(),"stage1_manifest_sha256":config["expected_inputs"]["stage1_manifest_sha256"],"stage2_config_sha256":config["expected_inputs"]["stage2_config_sha256"],"stage3_assignments_sha256":config["expected_inputs"]["stage3_assignments_sha256"],"stage4a_config_sha256":sha256_file("configs/kepler_stage4a_scale_v1.json"),"mast_inventory_sha256":sha256_file(root/"kepler_stage4a_mast_inventory_v1.csv"),"discovery_outcomes_sha256":sha256_file(root/"kepler_stage4a_discovery_outcomes_v1.csv"),"metadata_checksums_sha256":sha256_file(checksum_path),"fits_checksum_manifest_sha256":None,"processing_manifest_sha256":None,"array_checksum_manifest_sha256":None,"git_revision":revision,"worktree_dirty":dirty,"environment":{"python":sys.version.split()[0],"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"astropy":astropy.__version__,"astroquery":astroquery.__version__}}
 write_json(provenance,root/"kepler_stage4a_provenance_v1.json"); print(json.dumps(provenance,indent=2,sort_keys=True))
if __name__=="__main__": main()
