import pandas as pd
from kepler_scale.contracts import load_config
from kepler_scale.paths import resolve_data_paths
from kepler_scale.preflight import build_preflight
def test_laptop_free_space_uses_active_host_not_full_archive(tmp_path):
 c=load_config(); c["paths"]["require_external_to_repository"]=False; p=resolve_data_paths(str(tmp_path),c,repository_root=tmp_path,create=True); roster=pd.DataFrame({"kepid":[1,2],"koi_count":[1,1]}); batches=pd.DataFrame({"batch_id":["batch-00000"]}); inventory=pd.DataFrame({"kepid":[1,2],"product_filename":["a","b"],"product_size":[100,200]}); outcomes=pd.DataFrame({"status":["products_found","products_found"]}); report=build_preflight(c,"laptop_safe",p,roster,batches,inventory,outcomes); assert report["required_free_bytes"]==200+150*1024*1024+c["profiles"]["laptop_safe"]["safety_margin_bytes"]
