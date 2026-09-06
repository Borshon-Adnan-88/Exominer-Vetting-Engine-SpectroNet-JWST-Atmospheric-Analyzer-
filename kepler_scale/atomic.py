"""Same-filesystem atomic writes for synced and local filesystems."""

import json, os
from pathlib import Path


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); part=path.with_name(path.name+".part")
    with part.open("wb") as handle: handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(part,path)


def atomic_json(path: str | Path, value: dict) -> None:
    atomic_bytes(path,(json.dumps(value,indent=2,sort_keys=True)+"\n").encode())
