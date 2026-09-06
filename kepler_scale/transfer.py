"""Label-blind transfer-ceiling planning and verified resume ordering."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TransferPlan:
    kepids: tuple[int, ...]
    planned_bytes: int
    ceiling_bytes: int | None
    stop_reason: str
    next_incomplete_kepid: int | None


def plan_host_transfer(
    roster: pd.DataFrame,
    inventory: pd.DataFrame,
    completed_verified_hosts: set[int],
    max_download_bytes_per_run: int | None,
) -> TransferPlan:
    """Select complete hosts in stable roster order without crossing the ceiling."""
    if max_download_bytes_per_run is not None and max_download_bytes_per_run < 0:
        raise ValueError("max_download_bytes_per_run must be nonnegative or null")
    host_bytes = inventory.groupby("kepid", sort=False)["product_size"].sum().astype("int64").to_dict()
    incomplete = [int(k) for k in roster.sort_values("host_order", kind="stable")["kepid"] if int(k) not in completed_verified_hosts]
    selected: list[int] = []
    planned = 0
    for kepid in incomplete:
        required = int(host_bytes.get(kepid, 0))
        if max_download_bytes_per_run is not None and planned + required > max_download_bytes_per_run:
            return TransferPlan(tuple(selected), planned, max_download_bytes_per_run, "transfer_ceiling_reached", kepid)
        selected.append(kepid)
        planned += required
    return TransferPlan(tuple(selected), planned, max_download_bytes_per_run, "all_incomplete_hosts_planned", None)


def completed_verified_hosts(host_states: list[dict]) -> set[int]:
    """Resume only past hosts whose processing and checksums reached final verified state."""
    return {
        int(state["kepid"])
        for state in host_states
        if state.get("all_kois_final") is True
        and state.get("array_checksums_verified") is True
        and state.get("provenance_written") is True
    }
