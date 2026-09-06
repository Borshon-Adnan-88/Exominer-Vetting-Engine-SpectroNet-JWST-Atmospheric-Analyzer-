import pytest
from kepler_scale.purge import REQUIRED,assert_purge_eligible
def test_purge_requires_every_gate_and_contained_path(tmp_path):
 raw=tmp_path/"raw"; host=raw/"42"; host.mkdir(parents=True); (host/"x_llc.fits").write_bytes(b"x"); state={k:True for k in REQUIRED}
 with pytest.raises(PermissionError): assert_purge_eligible(state,host,raw,False)
 bad=dict(state); bad["provenance_written"]=False
 with pytest.raises(RuntimeError): assert_purge_eligible(bad,host,raw,True)
 assert assert_purge_eligible(state,host,raw,True)==[host/"x_llc.fits"]
