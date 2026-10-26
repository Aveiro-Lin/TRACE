#!/usr/bin/env python3
"""Export the patient-level TRACE paper split using only public identifiers."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


EXPECTED = {
    "train": {"records": 541, "patients": 538},
    "valid": {"records": 134, "patients": 134},
    "test": {"records": 677, "patients": 673},
    "excluded": {"records": 33, "patients": 32},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-metadata", type=Path, required=True)
    parser.add_argument("--locked-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    public_rows = read_csv(args.public_metadata)
    locked_rows = read_csv(args.locked_manifest)
    if len(public_rows) != 1385:
        raise ValueError(f"Expected 1385 public records, found {len(public_rows)}")
    if len(locked_rows) != 1352:
        raise ValueError(f"Expected 1352 eligible split rows, found {len(locked_rows)}")

    locked_by_uid: dict[str, dict[str, str]] = {}
    patient_partition: dict[str, str] = {}
    for row in locked_rows:
        uid = row["record_uid"]
        patient_id = row["patient_id"]
        partition = row["split"]
        if uid in locked_by_uid:
            raise ValueError(f"Duplicate record_uid in locked manifest: {uid}")
        if partition not in {"train", "valid", "test"}:
            raise ValueError(f"Unexpected partition {partition!r} for {uid}")
        previous = patient_partition.setdefault(patient_id, partition)
        if previous != partition:
            raise ValueError(f"Patient crosses partitions: {patient_id}")
        locked_by_uid[uid] = row

    public_uids = {row["record_uid"] for row in public_rows}
    unknown_uids = sorted(set(locked_by_uid).difference(public_uids))
    if unknown_uids:
        raise ValueError(f"Locked manifest contains non-public record IDs: {unknown_uids[:5]}")

    output_rows: list[dict[str, object]] = []
    for row_index, row in enumerate(public_rows):
        locked = locked_by_uid.get(row["record_uid"])
        partition = locked["split"] if locked else "excluded"
        output_rows.append(
            {
                "row_index": row_index,
                "ecg_id": row["ecg_id"],
                "record_uid": row["record_uid"],
                "patient_id": row["patient_id"],
                "paper_partition": partition,
                "eligible_for_report_control": "True" if locked else "False",
                "exclusion_reason": "" if locked else "no_mapped_diagnostic_form_or_rhythm_label",
                "split_unit": "patient_id",
                "seed": 42,
                "protocol": "50%_development_50%_locked_test_development_80%_train_20%_valid",
                "assignment_source": "locked_report_control_split_seed42",
            }
        )

    record_counts = Counter(str(row["paper_partition"]) for row in output_rows)
    patient_sets: dict[str, set[str]] = {
        partition: {
            str(row["patient_id"])
            for row in output_rows
            if row["paper_partition"] == partition
        }
        for partition in EXPECTED
    }
    for partition, expected in EXPECTED.items():
        observed = {
            "records": record_counts[partition],
            "patients": len(patient_sets[partition]),
        }
        if observed != expected:
            raise ValueError(f"Unexpected {partition} counts: {observed}; expected {expected}")

    fitted = ("train", "valid", "test")
    for left_index, left in enumerate(fitted):
        for right in fitted[left_index + 1 :]:
            overlap = patient_sets[left].intersection(patient_sets[right])
            if overlap:
                raise ValueError(f"Patient overlap between {left} and {right}: {sorted(overlap)[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Wrote {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
