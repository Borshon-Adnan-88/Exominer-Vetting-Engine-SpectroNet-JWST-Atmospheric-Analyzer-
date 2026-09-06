from kepler_scale.contracts import load_config,validate_inputs
def test_stage4a_frozen_input_contract():
 r=validate_inputs(load_config()); assert r=={**{k:v for k,v in r.items()},}; assert r["kois"]==6637 and r["hosts"]==5804 and r["assignment_rows"]==132740
