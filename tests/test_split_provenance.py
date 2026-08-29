import json
from kepler_splits.schemas import ASSIGNMENT_COLUMNS


def test_config_freezes_policy_and_forbidden_fields_are_absent():
    c=json.loads(open("configs/kepler_split_policy_object_vs_host_v1.json",encoding="utf-8").read())
    assert c["seeds"]==[1729,2718,3141,4099,5279,6553,7919,9341,104729,130363]
    assert "arbitrary deterministic" in c["algorithm"]["fold_id_note"]
    assert not set(c["forbidden_model_fields"]) & set(ASSIGNMENT_COLUMNS)
