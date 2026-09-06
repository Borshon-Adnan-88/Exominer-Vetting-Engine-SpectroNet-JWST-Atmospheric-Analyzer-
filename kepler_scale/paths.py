"""Location-independent external data-root resolution."""

import os
from dataclasses import dataclass
from pathlib import Path

from .contracts import ContractError


@dataclass(frozen=True)
class DataPaths:
    root: Path; raw_fits: Path; processed: Path; diagnostics: Path; cache: Path; temporary: Path


@dataclass(frozen=True)
class ScratchPaths:
    root: Path
    raw_fits: Path
    temporary: Path


def resolve_data_paths(cli_root: str | None, config: dict, repository_root: str | Path=".", create=False) -> DataPaths:
    supplied=cli_root or os.getenv(config["paths"]["environment_variable"])
    if not supplied: raise ContractError("set --data-root or EXOMINER_DATA_ROOT")
    root=Path(supplied).expanduser().resolve()/config["paths"]["dataset_subdirectory"]
    repo=Path(repository_root).resolve()
    if config["paths"]["require_external_to_repository"]:
        try: root.relative_to(repo); raise ContractError("Stage 4A data root must be outside the Git repository")
        except ValueError: pass
    paths=DataPaths(root,root/"raw_fits",root/"processed",root/"diagnostics",root/"cache",root/"temporary")
    if create:
        for value in paths.__dict__.values(): value.mkdir(parents=True,exist_ok=True)
    return paths


def relative_runtime_path(path: str | Path, paths: DataPaths) -> str:
    try: return Path(path).resolve().relative_to(paths.root).as_posix()
    except ValueError as exc: raise ContractError("runtime path is outside EXOMINER_DATA_ROOT") from exc


def resolve_scratch_paths(cli_root: str | None, data_paths: DataPaths, create: bool = False) -> ScratchPaths:
    """Resolve operational scratch; the result must never enter deterministic artifacts."""
    supplied = cli_root or os.getenv("EXOMINER_SCRATCH_ROOT")
    root = Path(supplied).expanduser().resolve() / "kepler_dr25" if supplied else data_paths.temporary
    paths = ScratchPaths(root=root, raw_fits=root / "raw_fits", temporary=root / "temporary")
    if create:
        paths.raw_fits.mkdir(parents=True, exist_ok=True)
        paths.temporary.mkdir(parents=True, exist_ok=True)
    return paths
