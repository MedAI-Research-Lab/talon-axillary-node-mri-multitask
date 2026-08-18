"""Audit a public release for PHI-like strings, local paths, and forbidden files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


FORBIDDEN_SUFFIXES = {".pt", ".pth", ".npz", ".npy", ".dcm", ".nrrd", ".docx", ".xlsx"}
FORBIDDEN_COLUMNS = {
    "maskedpatientname",
    "patientname",
    "originalpatientname",
    "patient_name",
}
PATTERNS = {
    "windows_user_path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),
    "masked_name": re.compile(r"\*{3,}"),
    "known_name_token": re.compile(
        r"sacide|z(?:ü|u)leyha|leman\s+atalay|alin\s+alp", re.I
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    findings: list[dict[str, object]] = []
    for path in args.root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.resolve() == Path(__file__).resolve() or path.name == "release_audit.json":
            continue
        relative = str(path.relative_to(args.root)).replace("\\", "/")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            findings.append(
                {"file": relative, "type": "forbidden_suffix", "detail": path.suffix}
            )
            continue
        if path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            bad = sorted(column for column in header if column.lower() in FORBIDDEN_COLUMNS)
            if bad:
                findings.append(
                    {"file": relative, "type": "forbidden_columns", "detail": bad}
                )
        if path.suffix.lower() in {
            ".py", ".md", ".txt", ".json", ".csv", ".toml", ".cff",
            ".drawio", ".svg", ".ipynb",
        }:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for label, pattern in PATTERNS.items():
                if pattern.search(text):
                    findings.append(
                        {"file": relative, "type": label, "detail": "pattern match"}
                    )
    report = {"root": ".", "passed": not findings, "findings": findings}
    report_path = args.json or args.root / "release_audit.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
