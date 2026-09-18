import json
from types import SimpleNamespace
import numpy as np,pandas as pd
from kepler_scale.contracts import load_config
from kepler_scale.operations import get_operational_profile
import pytest
from kepler_scale.download import DownloadError
from kepler_scale.runner import _ensure_host_products, run_batch, run_host
from kepler_scale.runner import _host_lock

def source_inventory(rows):
 return pd.DataFrame([{"kepid":42,"quarter":number+1,"product_filename":name,"data_uri":f"mast:{name}","product_size":len(payload),"expected_sha256":__import__('hashlib').sha256(payload).hexdigest()} for number,(name,payload) in enumerate(rows)])

def test_partial_host_download_fetches_only_missing_product(tmp_path,monkeypatch):
 raw=tmp_path/"42"; raw.mkdir(); existing=b"one"; missing=b"two"; (raw/"one.fits").write_bytes(existing); calls=[]; inventory=source_inventory([("one.fits",existing),("two.fits",missing)])
 def fetch(uri,destination,expected_size,config): calls.append(uri); destination.write_bytes(missing); return {"bytes":expected_size,"sha256":__import__('hashlib').sha256(missing).hexdigest(),"attempts":1}
 monkeypatch.setattr("kepler_scale.runner.download_atomic",fetch); _ensure_host_products(raw,inventory,load_config())
 assert calls==["mast:two.fits"]

def test_preprocessing_cannot_start_with_incomplete_host_set(tmp_path,monkeypatch):
 data=SimpleNamespace(root=tmp_path/"data",cache=tmp_path/"data"/"cache",temporary=tmp_path/"data"/"temporary"); scratch=SimpleNamespace(raw_fits=tmp_path/"scratch"/"raw_fits")
 inventory=source_inventory([("missing.fits",b"data")]); called=False
 monkeypatch.setattr("kepler_scale.runner.download_atomic",lambda *args,**kwargs:{"bytes":4,"sha256":"x","attempts":1})
 def read(*args,**kwargs):
  nonlocal called; called=True
 monkeypatch.setattr("kepler_scale.runner.read_kepler_quarter",read)
 with pytest.raises(DownloadError,match="source set incomplete"): run_host(42,inventory,pd.DataFrame(),load_config(),get_operational_profile("home_wifi"),data,scratch,False)
 assert not called

def test_stale_part_only_does_not_count_as_complete(tmp_path,monkeypatch):
 raw=tmp_path/"42"; raw.mkdir(); payload=b"data"; (raw/"x.fits.part").write_bytes(payload); calls=[]; inventory=source_inventory([("x.fits",payload)])
 def fetch(uri,destination,expected_size,config): calls.append(uri); destination.with_name(destination.name+".part").unlink(); destination.write_bytes(payload); return {"bytes":4,"sha256":__import__('hashlib').sha256(payload).hexdigest(),"attempts":1}
 monkeypatch.setattr("kepler_scale.runner.download_atomic",fetch); _ensure_host_products(raw,inventory,load_config())
 assert calls==["mast:x.fits"] and (raw/"x.fits").is_file() and not (raw/"x.fits.part").exists()

def test_verified_existing_product_is_not_redownloaded(tmp_path,monkeypatch):
 raw=tmp_path/"42"; raw.mkdir(); payload=b"data"; (raw/"x.fits").write_bytes(payload); inventory=source_inventory([("x.fits",payload)])
 monkeypatch.setattr("kepler_scale.runner.download_atomic",lambda *args,**kwargs:pytest.fail("verified file was re-downloaded")); ready=_ensure_host_products(raw,inventory,load_config())
 assert ready[0][2]["attempts"]==0

def test_batch_runner_uses_scratch_writes_data_purges_and_resumes(tmp_path,monkeypatch,kepler_fits_factory):
 data=SimpleNamespace(root=tmp_path/"data",processed=tmp_path/"data"/"processed",cache=tmp_path/"data"/"cache",temporary=tmp_path/"data"/"temporary",raw_fits=tmp_path/"data"/"raw_fits"); scratch=SimpleNamespace(root=tmp_path/"scratch",raw_fits=tmp_path/"scratch"/"raw_fits",temporary=tmp_path/"scratch"/"temporary")
 for p in (data.processed,data.cache,data.temporary,scratch.raw_fits,scratch.temporary): p.mkdir(parents=True,exist_ok=True)
 source=kepler_fits_factory(kepid=42,quarter=1); size=source.stat().st_size
 def fake(uri,destination,expected_size,config): destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(source.read_bytes()); return {"bytes":size,"sha256":__import__('hashlib').sha256(source.read_bytes()).hexdigest(),"attempts":1}
 monkeypatch.setattr("kepler_scale.runner.download_atomic",fake)
 roster=pd.DataFrame({"kepid":[42],"host_order":[0],"batch_id":["batch-00000"]}); inventory=pd.DataFrame({"kepid":[42],"quarter":[1],"product_filename":[source.name],"data_uri":["mast:x"],"product_size":[size]}); cohort=pd.DataFrame({"source_row_index":[1],"kepid":[42],"kepoi_name":["K00042.01"],"label":[1],"koi_period":[2.0],"koi_time0bk":[100.0],"koi_duration":[3.0]}); scientific=json.load(open("configs/kepler_lightcurve_preprocessing_v1.json")); scientific["retry"]=load_config()["retry"]
 first=run_batch("batch-00000",roster,inventory,cohort,scientific,get_operational_profile("home_wifi"),data,scratch); assert first["canonical_npz_count"]==1 and first["purged_hosts"]==1 and not list(scratch.raw_fits.rglob("*.fits"))
 second=run_batch("batch-00000",roster,inventory,cohort,scientific,get_operational_profile("home_wifi"),data,scratch); assert second["resumed_hosts"]==1 and second["downloaded_bytes"]==0


def test_host_lock_rejects_overlapping_writer_and_releases(tmp_path):
 data=SimpleNamespace(cache=tmp_path/"cache")
 with _host_lock(data,42):
  with pytest.raises(RuntimeError,match="already being processed"):
   with _host_lock(data,42): pytest.fail("second writer entered")
 with _host_lock(data,42): pass


def test_resume_reconciles_post_purge_residue_without_deleting_or_downloading(tmp_path,monkeypatch):
 data=SimpleNamespace(root=tmp_path/"data",cache=tmp_path/"data"/"cache")
 scratch=SimpleNamespace(raw_fits=tmp_path/"scratch"/"raw_fits")
 raw=scratch.raw_fits/"42"; raw.mkdir(parents=True); evidence=raw/"retained_llc.fits"; evidence.write_bytes(b"retain for audit")
 state={"kepid":42,"all_kois_final":True,"array_checksums_verified":True,"provenance_written":True,"raw_purged":True,"koi_outcomes":[]}
 state_path=data.cache/"host_state"/"000000042.json"; state_path.parent.mkdir(parents=True); state_path.write_text(json.dumps(state))
 monkeypatch.setattr("kepler_scale.runner.download_atomic",lambda *a,**k:pytest.fail("resume downloaded"))
 result,metric=run_host(42,pd.DataFrame(),pd.DataFrame(),{},get_operational_profile("home_wifi"),data,scratch,True)
 assert result["raw_purged"] is False and metric["resumed_skip"]
 assert result["raw_retention_reason"]=="residual_sources_detected_on_resume_requires_review"
 assert evidence.read_bytes()==b"retain for audit"
 assert json.loads(state_path.read_text())["raw_purged"] is False
