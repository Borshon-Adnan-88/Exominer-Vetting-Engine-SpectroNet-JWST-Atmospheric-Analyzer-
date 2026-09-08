import json
from types import SimpleNamespace
import numpy as np,pandas as pd
from kepler_scale.contracts import load_config
from kepler_scale.operations import get_operational_profile
from kepler_scale.runner import run_batch

def test_batch_runner_uses_scratch_writes_data_purges_and_resumes(tmp_path,monkeypatch,kepler_fits_factory):
 data=SimpleNamespace(root=tmp_path/"data",processed=tmp_path/"data"/"processed",cache=tmp_path/"data"/"cache",temporary=tmp_path/"data"/"temporary",raw_fits=tmp_path/"data"/"raw_fits"); scratch=SimpleNamespace(root=tmp_path/"scratch",raw_fits=tmp_path/"scratch"/"raw_fits",temporary=tmp_path/"scratch"/"temporary")
 for p in (data.processed,data.cache,data.temporary,scratch.raw_fits,scratch.temporary): p.mkdir(parents=True,exist_ok=True)
 source=kepler_fits_factory(kepid=42,quarter=1); size=source.stat().st_size
 def fake(uri,destination,expected_size,config): destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(source.read_bytes()); return {"bytes":size,"sha256":__import__('hashlib').sha256(source.read_bytes()).hexdigest(),"attempts":1}
 monkeypatch.setattr("kepler_scale.runner.download_atomic",fake)
 roster=pd.DataFrame({"kepid":[42],"host_order":[0],"batch_id":["batch-00000"]}); inventory=pd.DataFrame({"kepid":[42],"quarter":[1],"product_filename":[source.name],"data_uri":["mast:x"],"product_size":[size]}); cohort=pd.DataFrame({"source_row_index":[1],"kepid":[42],"kepoi_name":["K00042.01"],"label":[1],"koi_period":[2.0],"koi_time0bk":[100.0],"koi_duration":[3.0]}); scientific=json.load(open("configs/kepler_lightcurve_preprocessing_v1.json")); scientific["retry"]=load_config()["retry"]
 first=run_batch("batch-00000",roster,inventory,cohort,scientific,get_operational_profile("home_wifi"),data,scratch); assert first["canonical_npz_count"]==1 and first["purged_hosts"]==1 and not list(scratch.raw_fits.rglob("*.fits"))
 second=run_batch("batch-00000",roster,inventory,cohort,scientific,get_operational_profile("home_wifi"),data,scratch); assert second["resumed_hosts"]==1 and second["downloaded_bytes"]==0
