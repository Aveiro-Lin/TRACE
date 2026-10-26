#!/usr/bin/env python3
"""Regenerate the flat package manifest and SHA-256 checksum list."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    excluded = {"MANIFEST.csv", "SHA256SUMS.txt"}
    paths = sorted(
        path for path in root.iterdir() if path.is_file() and path.name not in excluded
    )
    rows = [
        {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    ]
    with (root / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(rows)
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{row['sha256']}  {row['file']}\n" for row in rows), encoding="utf-8"
    )
    print(f"Wrote checksums for {len(rows)} files")


if __name__ == "__main__":
    main()
