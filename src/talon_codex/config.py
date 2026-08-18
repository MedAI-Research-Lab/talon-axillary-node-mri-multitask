from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def experiment_run_directory(output_root: str | Path, experiment: str, seed: int) -> Path:
    """Return the canonical run directory, nesting ablations separately."""
    experiment_path = Path("ablations") / experiment if str(experiment).startswith("TALON_NO_") else Path(str(experiment))
    return Path(output_root) / "runs" / experiment_path / f"seed_{int(seed)}"


@dataclass(frozen=True)
class ResearchConfig:
    raw: dict[str, Any]
    source_path: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Configuration section {name!r} must be an object.")
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve portable config paths relative to the config file directory."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.source_path.parent / path
        return path.resolve()

    @property
    def output_root(self) -> Path:
        return self.resolve_path(self.raw["output_root"])

    @property
    def dataset_root(self) -> Path:
        return self.resolve_path(self.raw["dataset_root"])

    @property
    def metadata_path(self) -> Path:
        return self.resolve_path(self.raw["metadata_path"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def run_dir(self, experiment: str, seed: int) -> Path:
        return experiment_run_directory(self.output_root, experiment, seed)

    def snapshot(self, destination: Path, extra: Mapping[str, Any] | None = None) -> None:
        payload = dict(self.raw)
        if extra:
            payload["runtime"] = dict(extra)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_config(path: str | Path) -> ResearchConfig:
    config_path = Path(path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    required = ["dataset_root", "metadata_path", "output_root", "columns", "model", "loss", "training"]
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")
    return ResearchConfig(raw=raw, source_path=config_path)


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(dict(result[key]), value)
        else:
            result[key] = value
    return result
