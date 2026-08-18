"""Resume-safe sequential execution of seeds 123 and 2026 plus aggregation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
EXECUTION_ROOT = OUTPUT_ROOT / "execution"
LOG_ROOT = EXECUTION_ROOT / "logs" / "repeated_holdout"
STATUS_PATH = EXECUTION_ROOT / "all_remaining_seeds_status.json"
RUNNER = PROJECT_ROOT / "scripts" / "run_repeated_holdout.py"
AGGREGATOR = PROJECT_ROOT / "scripts" / "aggregate_repeated_holdout.py"
SEEDS = (123, 2026)
STAGES = (
    "prepare",
    "train-hybrid",
    "train-unet",
    "evaluate-hybrid",
    "evaluate-unet",
    "teacher-hybrid",
    "compare",
    "relative-overlap",
    "output-contract",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(state: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def completed_artifact(seed: int, stage: str) -> Path | None:
    hybrid = OUTPUT_ROOT / "runs" / "HYBRID_TALON" / f"seed_{seed}"
    unet = OUTPUT_ROOT / "runs" / "UNET_MTL_MASK_GUIDED" / f"seed_{seed}"
    mapping = {
        "prepare": PROJECT_ROOT / "data" / "splits" / f"seed_{seed}" / "split_integrity_audit.json",
        "train-hybrid": hybrid / "execution_training_status.json",
        "train-unet": unet / "execution_training_status.json",
        "evaluate-hybrid": hybrid / "execution_evaluation_status.json",
        "evaluate-unet": unet / "execution_evaluation_status.json",
        "teacher-hybrid": hybrid / "teacher_checkpoint_comparison" / "teacher_checkpoint_validation_comparison_status.json",
        "compare": OUTPUT_ROOT / "repeated_holdout_comparison" / "per_seed" / f"seed_{seed}" / "comparison_status.json",
        "relative-overlap": OUTPUT_ROOT / "repeated_holdout_comparison" / "per_seed" / f"seed_{seed}" / "relative_overlap_analysis" / "relative_overlap_analysis_metadata.json",
        "output-contract": OUTPUT_ROOT / "repeated_holdout_comparison" / "reproducibility" / f"output_contract_seed_{seed}.json",
    }
    return mapping.get(stage)


def artifact_is_complete(seed: int, stage: str) -> bool:
    path = completed_artifact(seed, stage)
    if path is None or not path.exists():
        return False
    if path.suffix.lower() != ".json":
        return True
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return str(payload.get("status", "")).lower() in {"passed", "completed"}


def initial_state() -> dict:
    if STATUS_PATH.exists():
        state = json.loads(STATUS_PATH.read_text(encoding="utf-8-sig"))
        if state.get("status") == "completed":
            return state
        state["status"] = "running"
        state["resumed_utc"] = utc_now()
        state.pop("error", None)
        state.pop("traceback", None)
        return state
    return {
        "status": "running",
        "started_utc": utc_now(),
        "python": sys.executable,
        "seeds": list(SEEDS),
        "seed_42_retrained": False,
        "stages": {
            str(seed): {stage: {"status": "pending"} for stage in STAGES}
            for seed in SEEDS
        },
        "aggregate": {"status": "pending"},
    }


def run_stage(state: dict, seed: int, stage: str) -> None:
    record = state["stages"][str(seed)][stage]
    if record.get("status") == "completed" or artifact_is_complete(seed, stage):
        record.update(
            {
                "status": "completed",
                "recovered_from_artifact": record.get("status") != "completed",
                "artifact": str(completed_artifact(seed, stage)),
            }
        )
        write_status(state)
        return

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"seed_{seed}_{stage}.log"
    record.update({"status": "running", "started_utc": utc_now(), "log": str(log_path)})
    state.update({"current_seed": seed, "current_stage": stage})
    write_status(state)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(
            [sys.executable, str(RUNNER), stage, "--seed", str(seed)],
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Seed {seed} stage {stage} failed; see {log_path}")
    if not artifact_is_complete(seed, stage):
        raise RuntimeError(f"Seed {seed} stage {stage} returned success but its completion artifact is absent.")
    record.update(
        {
            "status": "completed",
            "finished_utc": utc_now(),
            "elapsed_seconds": time.time() - started,
            "artifact": str(completed_artifact(seed, stage)),
        }
    )
    write_status(state)


def run_aggregate(state: dict) -> None:
    manifest = OUTPUT_ROOT / "repeated_holdout_comparison" / "reproducibility" / "aggregate_manifest.json"
    if state["aggregate"].get("status") == "completed" or manifest.exists():
        state["aggregate"].update({"status": "completed", "manifest": str(manifest)})
        write_status(state)
        return
    log_path = LOG_ROOT / "aggregate.log"
    state["aggregate"] = {"status": "running", "started_utc": utc_now(), "log": str(log_path)}
    state.update({"current_seed": None, "current_stage": "aggregate"})
    write_status(state)
    started = time.time()
    with log_path.open("a", encoding="utf-8") as stream:
        completed = subprocess.run(
            [sys.executable, str(AGGREGATOR)],
            cwd=PROJECT_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not manifest.exists():
        raise RuntimeError(f"Aggregate stage failed; see {log_path}")
    state["aggregate"].update(
        {
            "status": "completed",
            "finished_utc": utc_now(),
            "elapsed_seconds": time.time() - started,
            "manifest": str(manifest),
        }
    )
    write_status(state)


def main() -> None:
    state = initial_state()
    if state.get("status") == "completed":
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return
    write_status(state)
    try:
        for seed in SEEDS:
            for stage in STAGES:
                run_stage(state, seed, stage)
        run_aggregate(state)
        state.update(
            {
                "status": "completed",
                "finished_utc": utc_now(),
                "current_seed": None,
                "current_stage": None,
            }
        )
        write_status(state)
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        current_seed = state.get("current_seed")
        current_stage = state.get("current_stage")
        if current_seed is not None and current_stage in STAGES:
            state["stages"][str(current_seed)][current_stage]["status"] = "failed"
        elif current_stage == "aggregate":
            state["aggregate"]["status"] = "failed"
        write_status(state)
        raise


if __name__ == "__main__":
    main()
