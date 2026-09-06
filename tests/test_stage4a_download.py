from pathlib import Path
import pytest
from kepler_scale.contracts import load_config
from kepler_scale.download import download_atomic,DownloadError

def test_atomic_download_restarts_part_and_checks_size(tmp_path,monkeypatch):
 destination=tmp_path/"x.fits"; part=tmp_path/"x.fits.part"; part.write_bytes(b"old")
 def good(uri,local_path,cache): Path(local_path).write_bytes(b"data"); return "COMPLETE"
 monkeypatch.setattr("kepler_scale.download.Observations.download_file",good); result=download_atomic("mast:x",destination,4,load_config()); assert destination.read_bytes()==b"data" and not part.exists() and result["bytes"]==4

def test_retry_exhaustion_never_exposes_final_file(tmp_path,monkeypatch):
 c=load_config(); c["retry"]["backoff_seconds"]=[0,0,0,0]
 def short(uri,local_path,cache): Path(local_path).write_bytes(b"x"); return "COMPLETE"
 monkeypatch.setattr("kepler_scale.download.Observations.download_file",short)
 with pytest.raises(DownloadError,match="retry_exhausted"): download_atomic("mast:x",tmp_path/"x.fits",4,c)
 assert not (tmp_path/"x.fits").exists() and not (tmp_path/"x.fits.part").exists()
