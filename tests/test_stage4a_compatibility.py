import json,numpy as np
from kepler_lightcurves.fits import read_kepler_quarter
from kepler_lightcurves.preprocess import process_object
from kepler_scale.arrays import write_canonical_npz
def test_stage2_arrays_load_unchanged_in_canonical_writer(kepler_fits_factory,tmp_path):
 q=read_kepler_quarter(kepler_fits_factory(kepid=42,quarter=1),42); c=json.load(open("configs/kepler_lightcurve_preprocessing_v1.json")); arrays,_=process_object([q],2.0,100.0,3.0,c); write_canonical_npz(tmp_path/"x.npz",arrays)
 with np.load(tmp_path/"x.npz") as saved:
  for key,value in arrays.items(): np.testing.assert_array_equal(saved[key],value)
