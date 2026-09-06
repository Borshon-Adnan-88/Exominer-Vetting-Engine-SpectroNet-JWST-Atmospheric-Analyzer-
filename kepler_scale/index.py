"""Canonical KOI array identity and index validation."""

import re
import pandas as pd


def canonical_relative_path(kepid:int,kepoi_name:str)->str:
    safe=re.sub(r"[^A-Za-z0-9_-]","_",kepoi_name)
    return f"processed/{int(kepid)}/{safe}.npz"


def validate_unique_index(frame:pd.DataFrame)->None:
    if frame.kepoi_name.duplicated().any() or frame.relative_path.duplicated().any(): raise ValueError("duplicate canonical KOI output")
