#!/usr/bin/env python3
"""Create a deterministic public file manifest and SHA-256 list."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def role(path: Path) -> str:
    text = path.as_posix()
    if text.startswith("records500/"):
        return "wfdb_500hz_reconstruction_grid"
    if text.startswith("records100/"):
        return "wfdb_100hz_derived_grid"
    if text.startswith("validity_masks/"):
        return "observed_interval_mask"
    if text.startswith("trace_compat/"):
        return "trace_compatibility"
    if text.startswith("quality_control/"):
        return "quality_control"
    if text.startswith("processing_code/"):
        return "processing_code"
    return "metadata_or_documentation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    excluded = {"MANIFEST.csv", "SHA256SUMS.txt"}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in excluded
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )
    rows = []
    for index, path in enumerate(files, start=1):
        relative = path.relative_to(root)
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role(relative),
            }
        )
        if index % 500 == 0:
            print(f"hashed {index}/{len(files)} files", flush=True)

    with (root / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "size_bytes", "sha256", "role"))
        writer.writeheader()
        writer.writerows(rows)
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="utf-8"
    )
    print(f"wrote checksums for {len(rows)} files", flush=True)


if __name__ == "__main__":
    main()
