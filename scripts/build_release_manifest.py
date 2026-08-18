"""Create a deterministic SHA-256 manifest for all public-release files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {"SHA256SUMS.txt", "release_manifest.json", "release_audit.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or path.name in EXCLUDED
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or ".pytest_cache" in path.parts
            or ".qa_reproduced" in path.parts
        ):
            continue
        files.append(
            {
                "path": str(path.relative_to(ROOT)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (ROOT / "SHA256SUMS.txt").write_text(
        "\n".join(f"{item['sha256']}  {item['path']}" for item in files) + "\n",
        encoding="utf-8",
    )
    payload = {
        "release": "1.0.0",
        "seeds": [42, 123, 2026, 27182, 31415],
        "models": ["HYBRID_TALON", "UNET_MTL_MASK_GUIDED"],
        "file_count_excluding_manifest": len(files),
        "total_bytes_excluding_manifest": sum(item["bytes"] for item in files),
        "files": files,
    }
    (ROOT / "release_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
