"""Shared, locked helpers for the five-seed Hybrid TALON benchmark."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DATA_SPLIT_ROOT = PROJECT_ROOT / "data" / "splits"
EXECUTION_ROOT = OUTPUT_ROOT / "execution"
CANONICAL_HYBRID_ROOT = PROJECT_ROOT
CANONICAL_PARENT_ROOT = PROJECT_ROOT
CANONICAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "publication_config.frozen.json"
CANONICAL_HYBRID_MODEL_PATH = PROJECT_ROOT / "src" / "talon_publication" / "hybrid_talon.py"
FROZEN_HYBRID_MODEL_PATH = CANONICAL_HYBRID_MODEL_PATH
CORE_SOURCE_ROOT = PROJECT_ROOT / "src"
CANONICAL_HYBRID_PIPELINE_PATH = PROJECT_ROOT / "scripts" / "run_publication_pipeline.py"
SEEDS = (42, 123, 2026, 31415, 27182)
NEW_TRAINING_SEEDS = (123, 2026, 31415, 27182)
MODELS = ("HYBRID_TALON", "UNET_MTL_MASK_GUIDED")
EXPECTED_HYBRID_PARAMETERS = 4_930_849

for path in (str(CORE_SOURCE_ROOT), str(PROJECT_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from talon_codex.config import ResearchConfig  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def canonical_payload() -> dict[str, Any]:
    return json.loads(CANONICAL_CONFIG_PATH.read_text(encoding="utf-8"))


def config_for_seed(seed: int) -> ResearchConfig:
    """Clone seed-42 settings, changing only run identity and output root."""
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"Unsupported seed {seed}; expected one of {SEEDS}.")
    raw = copy.deepcopy(canonical_payload())
    raw["project_name"] = "Hybrid_TALON_Repeated_Holdout"
    raw["output_root"] = str(OUTPUT_ROOT)
    raw["seed"] = seed
    # build_data_bundle uses config.split_seed for the patient-grouped split.
    # Keeping this synchronized is essential: --seed alone is insufficient.
    raw["split_seed"] = seed
    raw["repeated_seeds"] = list(SEEDS)
    return ResearchConfig(raw=raw, source_path=PROJECT_ROOT / "configs" / f"runtime_seed_{seed}.json")


def normalized_scientific_config(payload: dict[str, Any]) -> dict[str, Any]:
    """Return settings that must remain identical across all repeated-holdout runs."""
    result = copy.deepcopy(payload)
    for key in ("project_name", "output_root", "seed", "split_seed", "repeated_seeds"):
        result.pop(key, None)
    return result


def scientific_config_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(normalized_scientific_config(payload), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_canonical_hybrid_pipeline() -> ModuleType:
    """Load the executed seed-42 pipeline and redirect only its output root."""
    spec = importlib.util.spec_from_file_location("canonical_hybrid_pipeline_repeated", CANONICAL_HYBRID_PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load canonical Hybrid pipeline: {CANONICAL_HYBRID_PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Import by an explicit unique module name.  The canonical pipeline imports
    # its own ``models`` package while loading, so a normal import here could be
    # satisfied from sys.modules by that external package instead of this frozen
    # byte-identical copy.
    frozen_spec = importlib.util.spec_from_file_location(
        "frozen_repeated_holdout_hybrid_talon", FROZEN_HYBRID_MODEL_PATH
    )
    if frozen_spec is None or frozen_spec.loader is None:
        raise ImportError(FROZEN_HYBRID_MODEL_PATH)
    frozen_module = importlib.util.module_from_spec(frozen_spec)
    frozen_spec.loader.exec_module(frozen_module)
    module.build_hybrid_talon = frozen_module.build_hybrid_talon
    module.PROJECT_ROOT = PROJECT_ROOT
    module.OUTPUT_ROOT = OUTPUT_ROOT
    module.EXECUTION_ROOT = EXECUTION_ROOT
    module.LOCKED_PARENT_OUTPUT = OUTPUT_ROOT
    module.CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime_generated.json"
    module.BASE.PROJECT_ROOT = PROJECT_ROOT
    module.BASE.OUTPUT_ROOT = OUTPUT_ROOT
    module.BASE.EXECUTION_ROOT = EXECUTION_ROOT
    module.BASE.CONFIG_PATH = module.CONFIG_PATH
    module.install_hybrid_dispatch()
    return module


def ensure_new_training_seed(seed: int) -> None:
    if int(seed) not in NEW_TRAINING_SEEDS:
        raise RuntimeError(
            f"Training/evaluation orchestration is restricted to {NEW_TRAINING_SEEDS}; "
            "seed_42 is an immutable copied reference run."
        )


def run_directory(model: str, seed: int) -> Path:
    if model not in MODELS:
        raise ValueError(f"Unsupported model: {model}")
    return OUTPUT_ROOT / "runs" / model / f"seed_{int(seed)}"


def seed_split_directory(seed: int) -> Path:
    return DATA_SPLIT_ROOT / f"seed_{int(seed)}"
