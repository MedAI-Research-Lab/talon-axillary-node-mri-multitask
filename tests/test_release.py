from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class ReleaseTests(unittest.TestCase):
    def test_five_seeds_and_two_models_are_present(self) -> None:
        path = ROOT / "results" / "aggregate" / "all_seed_model_metrics.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({42, 123, 2026, 27182, 31415}, {int(row["seed"]) for row in rows})
        self.assertEqual(
            {"HYBRID_TALON", "UNET_MTL_MASK_GUIDED"},
            {row["model"] for row in rows},
        )

    def test_scientific_seed_configuration(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "publication_config.frozen.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual([42, 123, 2026, 27182, 31415], config["repeated_seeds"])
        self.assertEqual(127, config["mask_threshold"])
        self.assertEqual(512, config["image_size"])

    def test_hybrid_parameter_count(self) -> None:
        import torch
        from talon_publication.hybrid_talon import build_hybrid_talon

        prior = torch.ones(1, 1, 512, 512)
        model = build_hybrid_talon({"base_channels": 32}, prior)
        count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        self.assertEqual(4_930_849, count)

    def test_clinical_release_warning_exists(self) -> None:
        text = (ROOT / "docs" / "DATA_GOVERNANCE.md").read_text(encoding="utf-8")
        self.assertIn("Mandatory check before public upload", text)


if __name__ == "__main__":
    unittest.main()
