import pandas as pd,pytest
from kepler_scale.index import canonical_relative_path,validate_unique_index
def test_canonical_path_and_duplicate_rejection():
 assert canonical_relative_path(42,"K00001.01")=="processed/42/K00001_01.npz"; x=pd.DataFrame({"kepoi_name":["a","a"],"relative_path":["x","y"]})
 with pytest.raises(ValueError): validate_unique_index(x)
