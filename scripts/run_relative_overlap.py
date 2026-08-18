"""Run the locked relative-overlap analysis for one completed model pair."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from benchmark_common import CANONICAL_HYBRID_ROOT, OUTPUT_ROOT, SEEDS, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    args = parser.parse_args()
    seed = int(args.seed)
    source = CANONICAL_HYBRID_ROOT / "scripts" / "run_relative_overlap_tiers.py"
    spec = importlib.util.spec_from_file_location(f"locked_relative_overlap_seed_{seed}", source)
    if spec is None or spec.loader is None:
        raise ImportError(source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = (
        OUTPUT_ROOT
        / "repeated_holdout_comparison"
        / "per_seed"
        / f"seed_{seed}"
        / "relative_overlap_analysis"
    )
    module.OUTPUT = destination
    module.MODELS = {
        "HYBRID_TALON": OUTPUT_ROOT / "runs" / "HYBRID_TALON" / f"seed_{seed}" / "evaluation",
        "UNET_MTL_MASK_GUIDED": OUTPUT_ROOT / "runs" / "UNET_MTL_MASK_GUIDED" / f"seed_{seed}" / "evaluation",
    }
    module.main()
    metadata_path = destination / "relative_overlap_analysis_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "seed": seed,
            "models": list(module.MODELS),
            "locked_analysis_source": str(source),
            "training_performed": False,
        }
    )
    write_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
