import pandas as pd
import pytest


def test_frozen_inventory_key_contract():
    inventory = pd.DataFrame({"kepid": [1, 1], "quarter": [1, 2], "data_uri": ["a", "b"]})
    assert not inventory.duplicated(["kepid", "quarter"]).any()
    inventory.loc[1, "quarter"] = 1
    assert inventory.duplicated(["kepid", "quarter"]).any()
