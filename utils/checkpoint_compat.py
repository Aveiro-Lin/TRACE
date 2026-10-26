"""Compatibility helpers for checkpoints created before public API renaming."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def remap_legacy_foundation_encoder_keys(
    state_dict: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    """Rename the legacy ``stmem_encoder`` path without relaxing strict loads."""

    remapped: dict[str, Any] = {}
    source_keys: dict[str, str] = {}
    renamed = 0
    for key, value in state_dict.items():
        parts = key.split(".")
        normalized = ".".join(
            "foundation_encoder" if part == "stmem_encoder" else part
            for part in parts
        )
        if normalized in remapped and source_keys[normalized] != key:
            raise ValueError(
                f"Checkpoint key collision while remapping {key!r} to {normalized!r}"
            )
        remapped[normalized] = value
        source_keys[normalized] = key
        renamed += int(normalized != key)
    return remapped, renamed
