"""Static two-pass parity audit; performs no training or model inference."""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from benchmark_common import (
    CANONICAL_CONFIG_PATH,
    CANONICAL_HYBRID_MODEL_PATH,
    CORE_SOURCE_ROOT,
    EXPECTED_HYBRID_PARAMETERS,
    FROZEN_HYBRID_MODEL_PATH,
    MODELS,
    NEW_TRAINING_SEEDS,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    canonical_payload,
    config_for_seed,
    scientific_config_hash,
    sha256,
    write_json,
)
from models.hybrid_talon import build_hybrid_talon
from talon_codex.models import (
    MaskGuidedMultiTaskUNet,
    baseline_width_from_checkpoint,
    build_capacity_matched_baseline,
    build_talon,
)


def state_schema(state: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(value.shape) for key, value in state.items()}


def checkpoint_state(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload, payload["model_state_dict"]


def hybrid_prior_from_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    candidates = [value for key, value in state.items() if key.endswith("dataset_prior.prior")]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one Hybrid dataset prior tensor, found {len(candidates)}")
    return candidates[0].detach().clone()


def audit(pass_id: int) -> dict[str, object]:
    canonical = canonical_payload()
    seed_configs = {seed: config_for_seed(seed) for seed in NEW_TRAINING_SEEDS}
    canonical_hash = scientific_config_hash(canonical)
    config_checks = {
        str(seed): {
            "scientific_hash": scientific_config_hash(config.raw),
            "scientific_settings_identical": scientific_config_hash(config.raw) == canonical_hash,
            "seed": config.seed,
            "split_seed": int(config.get("split_seed")),
            "seed_and_split_seed_match": config.seed == int(config.get("split_seed")),
            "output_root": str(config.output_root),
        }
        for seed, config in seed_configs.items()
    }

    hybrid_checkpoint = OUTPUT_ROOT / "runs" / "HYBRID_TALON" / "seed_42" / "checkpoints" / "best_overall.pt"
    unet_checkpoint = OUTPUT_ROOT / "runs" / "UNET_MTL_MASK_GUIDED" / "seed_42" / "checkpoints" / "best_overall.pt"
    hybrid_payload, hybrid_state = checkpoint_state(hybrid_checkpoint)
    unet_payload, unet_state = checkpoint_state(unet_checkpoint)

    hybrid_model = build_hybrid_talon(canonical["model"], hybrid_prior_from_state(hybrid_state))
    hybrid_load = hybrid_model.load_state_dict(hybrid_state, strict=True)
    hybrid_parameters = sum(parameter.numel() for parameter in hybrid_model.parameters() if parameter.requires_grad)

    unet_width = baseline_width_from_checkpoint(unet_payload, canonical["model"])
    unet_model = MaskGuidedMultiTaskUNet(unet_width, float(canonical["model"]["classifier_dropout"]))
    unet_load = unet_model.load_state_dict(unet_state, strict=True)
    unet_parameters = sum(parameter.numel() for parameter in unet_model.parameters() if parameter.requires_grad)
    # Re-run only the constructor-level capacity selection (no forward pass) to
    # prove that the new-seed training branch will build the same width-36 U-Net.
    reference_talon = build_talon(canonical["model"], hybrid_prior_from_state(hybrid_state), "TALON_FULL")
    newly_built_unet, capacity_rows = build_capacity_matched_baseline(canonical["model"], reference_talon)
    newly_built_unet_parameters = sum(
        parameter.numel() for parameter in newly_built_unet.parameters() if parameter.requires_grad
    )
    new_unet_schema = state_schema(newly_built_unet.state_dict())

    runner_path = PROJECT_ROOT / "scripts" / "run_repeated_holdout.py"
    ast.parse(runner_path.read_text(encoding="utf-8"), filename=str(runner_path))
    source_text = runner_path.read_text(encoding="utf-8")
    runner_contract = {
        "split_seed_explicitly_synchronized": 'raw["split_seed"] = seed' in (PROJECT_ROOT / "scripts" / "benchmark_common.py").read_text(encoding="utf-8"),
        "seed_42_training_forbidden": "ensure_new_training_seed" in source_text,
        "hybrid_and_unet_present": all(model in source_text for model in MODELS),
        "teacher_after_hybrid_evaluation": source_text.index('"teacher-hybrid"') > source_text.index('"evaluate-hybrid"'),
        "same_pair_compared": "stage_compare" in source_text,
    }

    source_hashes = {
        "canonical_config": sha256(CANONICAL_CONFIG_PATH),
        "canonical_hybrid_model": sha256(CANONICAL_HYBRID_MODEL_PATH),
        "frozen_hybrid_model": sha256(FROZEN_HYBRID_MODEL_PATH),
        "training_core": sha256(CORE_SOURCE_ROOT / "talon_codex" / "training.py"),
        "data_core": sha256(CORE_SOURCE_ROOT / "talon_codex" / "data.py"),
        "evaluation_core": sha256(CORE_SOURCE_ROOT / "talon_codex" / "evaluation.py"),
    }
    checks = {
        "frozen_hybrid_source_byte_identical": source_hashes["canonical_hybrid_model"] == source_hashes["frozen_hybrid_model"],
        "hybrid_seed42_checkpoint_strict_load": not hybrid_load.missing_keys and not hybrid_load.unexpected_keys,
        "hybrid_parameter_count_identical": hybrid_parameters == EXPECTED_HYBRID_PARAMETERS,
        "hybrid_checkpoint_identity": hybrid_payload.get("experiment") == "HYBRID_TALON" and int(hybrid_payload.get("seed")) == 42,
        "unet_seed42_checkpoint_strict_load": not unet_load.missing_keys and not unet_load.unexpected_keys,
        "unet_checkpoint_identity": unet_payload.get("experiment") == "UNET_MTL_MASK_GUIDED" and int(unet_payload.get("seed")) == 42,
        "new_seed_unet_constructor_parameter_count_identical": newly_built_unet_parameters == unet_parameters,
        "new_seed_unet_constructor_state_schema_identical": new_unet_schema == state_schema(unet_state),
        "all_seed_scientific_configs_identical": all(item["scientific_settings_identical"] for item in config_checks.values()),
        "all_seed_split_seeds_correct": all(item["seed_and_split_seed_match"] for item in config_checks.values()),
        "runner_contract_complete": all(runner_contract.values()),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pass_id": pass_id,
        "training_performed": False,
        "forward_inference_performed": False,
        "canonical_scientific_config_hash": canonical_hash,
        "seed_configs": config_checks,
        "source_hashes": source_hashes,
        "hybrid": {
            "seed_42_checkpoint": str(hybrid_checkpoint),
            "trainable_parameters": hybrid_parameters,
            "state_entries": len(state_schema(hybrid_state)),
            "strict_load": checks["hybrid_seed42_checkpoint_strict_load"],
        },
        "unet": {
            "seed_42_checkpoint": str(unet_checkpoint),
            "base_width": int(unet_width),
            "trainable_parameters": unet_parameters,
            "new_seed_constructor_trainable_parameters": newly_built_unet_parameters,
            "capacity_matching_rows": capacity_rows,
            "state_entries": len(state_schema(unet_state)),
            "strict_load": checks["unet_seed42_checkpoint_strict_load"],
        },
        "runner_contract": runner_contract,
        "checks": checks,
    }
    destination = OUTPUT_ROOT / "repeated_holdout_comparison" / "reproducibility" / f"seed_parity_audit_pass_{pass_id}.json"
    write_json(destination, result)
    if result["status"] != "passed":
        raise RuntimeError(f"Seed parity audit pass {pass_id} failed: {json.dumps(checks, indent=2)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass-id", type=int, required=True, choices=(1, 2, 3, 4))
    args = parser.parse_args()
    print(json.dumps(audit(args.pass_id), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
