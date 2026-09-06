"""Atomic per-host operational checkpoint records."""

import json
from pathlib import Path
from .atomic import atomic_json


def checkpoint_path(cache: str | Path, kepid: int) -> Path: return Path(cache)/"discovery"/f"{int(kepid):09d}.json"
def read_checkpoint(cache: str | Path,kepid:int):
    path=checkpoint_path(cache,kepid)
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError): return None
def write_checkpoint(cache: str | Path,kepid:int,value:dict): atomic_json(checkpoint_path(cache,kepid),value)
