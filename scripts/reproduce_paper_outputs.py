"""Regenerate public aggregate tables and the five-seed performance figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2026, 27182, 31415]
MODELS = {
    "HYBRID_TALON": ("TALON", "#234f7d", "o"),
    "UNET_MTL_MASK_GUIDED": ("Mask-guided U-Net", "#c65a3a", "s"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(ROOT / "results" / "aggregate" / "all_seed_model_metrics.csv")
    calibrated = metrics.query("calibration == 'calibrated'").copy()
    panels = [
        ("auroc", "AUROC", (0.0, 1.0)),
        ("auprc", "AUPRC", (0.0, 1.0)),
        ("mean_dice", "Dice", (0.0, 1.0)),
        ("brier", "Brier score (lower is better)", (0.0, 0.35)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, (column, title, limits) in zip(axes.flat, panels):
        for model, (label, color, marker) in MODELS.items():
            data = calibrated.query("model == @model").set_index("seed").loc[SEEDS]
            ax.plot(
                [str(seed) for seed in SEEDS],
                data[column],
                marker=marker,
                color=color,
                label=label,
                linewidth=2,
            )
        ax.set_title(title)
        ax.set_ylim(*limits)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.savefig(
        args.output_dir / "figure_2_five_seed_performance_reproduced.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    aggregate = pd.read_csv(ROOT / "results" / "aggregate" / "five_seed_mean_sd_range.csv")
    paired = pd.read_csv(
        ROOT / "results" / "paired_comparisons" / "five_seed_paired_effect_summary.csv"
    )
    aggregate.to_csv(args.output_dir / "table_1_five_seed_summary_reproduced.csv", index=False)
    paired.to_csv(args.output_dir / "table_2_paired_effects_reproduced.csv", index=False)
    print(f"Wrote public outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
