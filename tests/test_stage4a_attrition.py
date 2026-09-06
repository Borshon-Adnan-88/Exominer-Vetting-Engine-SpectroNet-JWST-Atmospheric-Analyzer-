import pandas as pd,pytest
from kepler_scale.attrition import validate_outcomes,build_attrition
def test_attrition_keeps_original_denominator():
 c=pd.DataFrame({"source_row_index":[1,2],"kepid":[1,2],"kepoi_name":["a","b"],"label":[0,1]}); o=c.copy(); o["technical_status"]=["constructed_qc_fail","retrieval_failure"]; o["model_data_available"]=[True,False]; validate_outcomes(o,c); a=pd.concat([c.assign(strategy="x",repeat_id=0,partition="train"),c.assign(strategy="y",repeat_id=0,partition="test")]); assert build_attrition(a,o).rows.sum()==4
