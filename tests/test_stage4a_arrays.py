import zipfile,numpy as np
from kepler_scale.arrays import KEYS,write_canonical_npz
def test_whole_npz_and_logical_hashes_are_deterministic(tmp_path):
 arrays={"global_count":np.arange(2000,dtype=np.int32),"global_flux":np.ones(2000,dtype=np.float32),"global_observed_mask":np.ones(2000,dtype=bool),"local_count":np.arange(200,dtype=np.int32),"local_flux":np.ones(200,dtype=np.float32),"local_observed_mask":np.ones(200,dtype=bool)}
 a=write_canonical_npz(tmp_path/"a.npz",arrays); b=write_canonical_npz(tmp_path/"b.npz",arrays); assert a==b
 with zipfile.ZipFile(tmp_path/"a.npz") as z: assert [x.filename[:-4] for x in z.infolist()]==list(KEYS) and all(x.date_time==(1980,1,1,0,0,0) and x.compress_type==zipfile.ZIP_STORED for x in z.infolist())
