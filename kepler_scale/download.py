"""Atomic, bounded-retry download primitives (not invoked at the preflight checkpoint)."""

import hashlib, os, time
from pathlib import Path
from astroquery.mast import Observations


class DownloadError(RuntimeError): pass


def download_atomic(data_uri:str,destination:Path,expected_size:int,config:dict)->dict:
    destination=Path(destination); part=destination.with_name(destination.name+".part")
    for attempt in range(1,config["retry"]["maximum_attempts"]+1):
        try:
            destination.parent.mkdir(parents=True,exist_ok=True)
            if part.exists(): part.unlink()
            status=Observations.download_file(data_uri,local_path=str(part),cache=False); name=status[0] if isinstance(status,tuple) else status
            if str(name).upper() not in {"COMPLETE","LOCAL"}: raise DownloadError(f"MAST status {status}")
            if not part.exists() or part.stat().st_size!=int(expected_size): raise DownloadError("advertised_size_mismatch")
            digest=hashlib.sha256(part.read_bytes()).hexdigest()
            os.replace(part,destination)
            return {"relative_path":None,"bytes":int(expected_size),"sha256":digest,"attempts":attempt}
        except (OSError,DownloadError) as exc:
            if part.exists(): part.unlink()
            if attempt==config["retry"]["maximum_attempts"]: raise DownloadError(f"retry_exhausted: {exc}") from exc
            time.sleep(config["retry"]["backoff_seconds"][min(attempt-1,len(config["retry"]["backoff_seconds"])-1)])
    raise AssertionError("unreachable")
