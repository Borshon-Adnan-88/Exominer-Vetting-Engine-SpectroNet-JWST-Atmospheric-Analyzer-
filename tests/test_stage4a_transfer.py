import pandas as pd

from kepler_scale.transfer import completed_verified_hosts, plan_host_transfer


def fixtures():
    roster = pd.DataFrame({"kepid": [10, 20, 30], "host_order": [0, 1, 2]})
    inventory = pd.DataFrame({"kepid": [10, 20, 30], "product_size": [40, 70, 20]})
    return roster, inventory


def test_transfer_ceiling_stops_before_exceeding_next_host():
    roster, inventory = fixtures()
    plan = plan_host_transfer(roster, inventory, set(), 100)
    assert plan.kepids == (10,)
    assert plan.planned_bytes == 40 and plan.next_incomplete_kepid == 20
    assert plan.stop_reason == "transfer_ceiling_reached"


def test_resume_starts_from_next_incomplete_verified_host():
    roster, inventory = fixtures()
    states = [
        {"kepid": 10, "all_kois_final": True, "array_checksums_verified": True, "provenance_written": True},
        {"kepid": 20, "all_kois_final": True, "array_checksums_verified": False, "provenance_written": True},
    ]
    plan = plan_host_transfer(roster, inventory, completed_verified_hosts(states), 100)
    assert plan.kepids == (20, 30) and plan.planned_bytes == 90


def test_zero_ceiling_stops_cleanly_without_partial_host():
    roster, inventory = fixtures()
    plan = plan_host_transfer(roster, inventory, set(), 0)
    assert plan.kepids == () and plan.planned_bytes == 0 and plan.next_incomplete_kepid == 10
