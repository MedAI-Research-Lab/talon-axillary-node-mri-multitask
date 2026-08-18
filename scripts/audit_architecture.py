"""Verify the frozen publication TALON parameter count without loading data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from talon_publication.hybrid_talon import build_hybrid_talon  # noqa: E402


EXPECTED_TRAINABLE_PARAMETERS = 4_930_849


def main() -> None:
    model = build_hybrid_talon({"base_channels": 32}, torch.ones(1, 1, 512, 512))
    observed = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = {
        "status": "passed" if observed == EXPECTED_TRAINABLE_PARAMETERS else "failed",
        "model": "TALON",
        "internal_run_id": "HYBRID_TALON",
        "expected_trainable_parameters": EXPECTED_TRAINABLE_PARAMETERS,
        "observed_trainable_parameters": observed,
        "all_required_audits_pass": observed == EXPECTED_TRAINABLE_PARAMETERS,
    }
    path = ROOT / "reports" / "architecture_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_required_audits_pass"] else 1)


if __name__ == "__main__":
    main()
