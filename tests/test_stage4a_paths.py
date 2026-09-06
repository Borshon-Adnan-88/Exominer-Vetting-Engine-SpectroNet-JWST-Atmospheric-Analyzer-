from pathlib import Path
import pytest
from kepler_scale.contracts import load_config,ContractError
from kepler_scale.paths import resolve_data_paths,relative_runtime_path
def test_cli_root_precedes_environment_and_paths_are_relative(tmp_path,monkeypatch):
 c=load_config(); monkeypatch.setenv("EXOMINER_DATA_ROOT",str(tmp_path/"env")); p=resolve_data_paths(str(tmp_path/"cli"),c,repository_root=tmp_path/"repo",create=True); assert p.root== (tmp_path/"cli"/"kepler_dr25").resolve(); assert relative_runtime_path(p.processed/"1"/"x.npz",p)=="processed/1/x.npz"
def test_missing_or_repository_internal_root_rejected(tmp_path,monkeypatch):
 c=load_config(); monkeypatch.delenv("EXOMINER_DATA_ROOT",raising=False)
 with pytest.raises(ContractError): resolve_data_paths(None,c,tmp_path)
 with pytest.raises(ContractError): resolve_data_paths(str(tmp_path/"inside"),c,tmp_path)
