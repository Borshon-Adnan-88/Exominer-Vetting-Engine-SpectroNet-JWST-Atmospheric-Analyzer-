"""Cross-platform byte-deterministic canonical NPZ output."""
import hashlib,io,os,zipfile
from pathlib import Path
import numpy as np

KEYS=("global_count","global_flux","global_observed_mask","local_count","local_flux","local_observed_mask")

def canonical_array(array:np.ndarray)->np.ndarray:
    value=np.asarray(array)
    if value.dtype.byteorder not in ("|","<") or (value.dtype.byteorder=="=" and not np.little_endian): value=value.astype(value.dtype.newbyteorder("<"),copy=False)
    return np.ascontiguousarray(value)

def logical_hash(array:np.ndarray)->str:
    value=canonical_array(array); header=f"{value.dtype.str}|{','.join(map(str,value.shape))}|".encode()
    return hashlib.sha256(header+value.tobytes(order="C")).hexdigest()

def write_canonical_npz(path:str|Path,arrays:dict[str,np.ndarray])->dict:
    if set(arrays)!=set(KEYS): raise ValueError("canonical NPZ keys mismatch")
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); part=path.with_name(path.name+".part")
    hashes={}
    with zipfile.ZipFile(part,"w",compression=zipfile.ZIP_STORED) as archive:
        for key in KEYS:
            value=canonical_array(arrays[key]); hashes[key]=logical_hash(value); buffer=io.BytesIO(); np.lib.format.write_array(buffer,value,allow_pickle=False)
            info=zipfile.ZipInfo(f"{key}.npy",date_time=(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_STORED; info.external_attr=0o600<<16; archive.writestr(info,buffer.getvalue())
    with part.open("r+b") as handle: handle.flush(); os.fsync(handle.fileno())
    os.replace(part,path); hashes["npz_sha256"]=hashlib.sha256(path.read_bytes()).hexdigest(); return hashes
