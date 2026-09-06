import json,re
from pathlib import Path
def test_deterministic_stage4a_files_have_no_machine_paths():
 paths=[Path("configs/kepler_stage4a_scale_v1.json"),Path("configs/kepler_stage4a_operational_profiles_v1.json"),*Path("data/stage4a").glob("*")]
 forbidden=re.compile(r"[A-Za-z]:[\\/]|/content/drive|/scratch/|Users[/\\]")
 for path in paths:
  if path.is_file(): assert not forbidden.search(path.read_text(encoding="utf-8"))
