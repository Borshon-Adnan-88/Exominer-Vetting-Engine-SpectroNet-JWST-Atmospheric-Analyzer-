import pandas as pd
import pytest
from kepler_scale.discovery import freeze_discovery
def record(k,status="products_found",q=1): return {"kepid":k,"status":status,"attempts":1,"error":None,"stage4a_config_sha256":"x","products":[] if status!="products_found" else [{"kepid":k,"quarter":q,"obsid":str(k),"product_filename":f"kplr{k:09d}-x_llc.fits","data_uri":f"mast:{k}:{q}","product_size":10,"description":f"Q{q}"}]}
def test_freeze_is_sorted_and_includes_no_product_outcome():
 i,o=freeze_discovery([record(2),record(1,"no_archive_product")]); assert i.kepid.tolist()==[2] and o.kepid.tolist()==[1,2]
def test_duplicate_target_quarter_is_rejected():
 r=record(2); r["products"].append({**r["products"][0],"data_uri":"other"})
 with pytest.raises(RuntimeError,match="ambiguous"): freeze_discovery([r])
