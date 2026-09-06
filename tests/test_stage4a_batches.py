import pandas as pd
from kepler_scale.batches import build_roster
def test_label_blind_deterministic_batches():
 c=pd.DataFrame({"kepid":[9,2,2,5],"kepoi_name":["a","b","c","d"],"label":[0,1,0,1]}); r,b=build_roster(c,2); assert r.kepid.tolist()==[2,5,9]; assert r.batch_id.tolist()==["batch-00000","batch-00000","batch-00001"]; assert "label" not in r
