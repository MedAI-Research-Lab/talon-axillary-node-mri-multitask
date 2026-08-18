"""Aggregate all paired repeated-holdout runs without pseudo-replication."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from benchmark_common import OUTPUT_ROOT, SEEDS, write_json


PER_SEED = OUTPUT_ROOT / "repeated_holdout_comparison" / "per_seed"
AGGREGATE = OUTPUT_ROOT / "repeated_holdout_comparison" / "aggregate_metrics"
PAIRED = OUTPUT_ROOT / "repeated_holdout_comparison" / "paired_comparisons"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    summaries: list[pd.DataFrame] = []
    paired: list[pd.DataFrame] = []
    inputs: list[dict[str, object]] = []
    for seed in SEEDS:
        root = PER_SEED / f"seed_{seed}"
        summary_path = root / "hybrid_vs_unet_summary.csv"
        paired_path = root / "paired_statistical_comparisons.csv"
        if not summary_path.exists() or not paired_path.exists():
            raise FileNotFoundError(f"Seed {seed} comparison is incomplete: {root}")
        summary = pd.read_csv(summary_path)
        summary.insert(0, "seed", seed)
        summaries.append(summary)
        pair = pd.read_csv(paired_path)
        pair.insert(0, "seed", seed)
        paired.append(pair)
        inputs.extend(
            [
                {"seed": seed, "path": str(summary_path), "sha256": file_hash(summary_path)},
                {"seed": seed, "path": str(paired_path), "sha256": file_hash(paired_path)},
            ]
        )

    per_seed_summary = pd.concat(summaries, ignore_index=True)
    per_seed_paired = pd.concat(paired, ignore_index=True)
    numeric = [
        column
        for column in per_seed_summary.select_dtypes(include="number").columns
        if column not in {"seed", "cases"}
    ]
    aggregate = (
        per_seed_summary.groupby(["model", "calibration"], dropna=False)[numeric]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else str(column)
        for column in aggregate.columns
    ]

    effect_column = "difference" if "difference" in per_seed_paired else "effect_a_minus_b"
    paired_summary = (
        per_seed_paired.groupby(["endpoint", "method"], dropna=False)[effect_column]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={effect_column: "effect"})
    )

    AGGREGATE.mkdir(parents=True, exist_ok=True)
    PAIRED.mkdir(parents=True, exist_ok=True)
    seed_count_words = {3: "three", 5: "five"}
    count_label = seed_count_words.get(len(SEEDS), str(len(SEEDS)))
    per_seed_summary.to_csv(AGGREGATE / "all_seed_model_metrics.csv", index=False)
    aggregate.to_csv(AGGREGATE / f"{count_label}_seed_mean_sd_range.csv", index=False)
    per_seed_paired.to_csv(PAIRED / "all_seed_paired_results.csv", index=False)
    paired_summary.to_csv(PAIRED / f"{count_label}_seed_paired_effect_summary.csv", index=False)
    write_json(
        OUTPUT_ROOT / "repeated_holdout_comparison" / "reproducibility" / "aggregate_manifest.json",
        {
            "status": "completed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": list(SEEDS),
            "aggregation_unit": "seed-level metric",
            "pseudo_replication_avoided": True,
            "note": "Repeated test appearances are not pooled as independent patients.",
            "inputs": inputs,
        },
    )


if __name__ == "__main__":
    main()
