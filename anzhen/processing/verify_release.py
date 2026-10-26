#!/usr/bin/env python3
"""Verify integrity, privacy minimization and TRACE interoperability."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb


LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def text_files(root: Path) -> list[Path]:
    allowed = {".csv", ".json", ".md", ".txt", ".hea", ".cff", ".py", ""}
    return [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed]


def main() -> int:
    args = parse_args()
    public_root = args.public_root.resolve()
    private_root = args.private_root.resolve()
    report_path = (args.report or public_root / "quality_control" / "release_validation.json").resolve()
    checks: list[dict[str, Any]] = []

    metadata = pd.read_csv(public_root / "anzhen_database.csv")
    mapping = pd.read_csv(private_root / "source_to_public_mapping.csv", dtype=str).fillna("")
    row_index = pd.read_csv(public_root / "trace_compat" / "row_index.csv")
    paper_split = pd.read_csv(public_root / "trace_compat" / "paper_report_control_split_seed42.csv")
    statements = pd.read_csv(public_root / "trace_compat" / "scp_statements.csv", index_col=0)
    n_records = len(metadata)

    add_check(checks, "nonempty_dataset", n_records > 0, n_records)
    expected_ids = list(range(1, n_records + 1))
    add_check(
        checks,
        "sequential_unique_ecg_ids",
        metadata.ecg_id.astype(int).tolist() == expected_ids,
        {"first": int(metadata.ecg_id.min()), "last": int(metadata.ecg_id.max())},
    )
    add_check(
        checks,
        "opaque_patient_ids",
        bool(metadata.patient_id.astype(str).str.fullmatch(r"AZP[A-F0-9]{16}").all()),
        int(metadata.patient_id.nunique()),
    )
    add_check(
        checks,
        "opaque_record_uids",
        bool(metadata.record_uid.astype(str).str.fullmatch(r"AZE[A-F0-9]{16}").all()),
        int(metadata.record_uid.nunique()),
    )
    add_check(
        checks,
        "metadata_mapping_counts_match",
        len(mapping) == n_records and len(row_index) == n_records,
        {"metadata": n_records, "private_mapping": len(mapping), "row_index": len(row_index)},
    )
    split_record_counts = paper_split.paper_partition.value_counts(dropna=False).to_dict()
    split_patient_counts = (
        paper_split.groupby("paper_partition").patient_id.nunique().to_dict()
    )
    expected_split_records = {"test": 677, "train": 541, "valid": 134, "excluded": 33}
    expected_split_patients = {"test": 673, "train": 538, "valid": 134, "excluded": 32}
    split_ids_match = (
        paper_split.record_uid.astype(str).tolist() == metadata.record_uid.astype(str).tolist()
        and paper_split.patient_id.astype(str).tolist() == metadata.patient_id.astype(str).tolist()
        and paper_split.ecg_id.astype(int).tolist() == metadata.ecg_id.astype(int).tolist()
    )
    patient_sets = {
        partition: set(
            paper_split.loc[paper_split.paper_partition == partition, "patient_id"].astype(str)
        )
        for partition in ("train", "valid", "test")
    }
    patient_overlap = {
        "train_valid": len(patient_sets["train"] & patient_sets["valid"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "valid_test": len(patient_sets["valid"] & patient_sets["test"]),
    }
    add_check(
        checks,
        "paper_report_control_split",
        len(paper_split) == n_records
        and split_ids_match
        and split_record_counts == expected_split_records
        and split_patient_counts == expected_split_patients
        and not any(patient_overlap.values()),
        {
            "records": split_record_counts,
            "patients": split_patient_counts,
            "patient_overlap": patient_overlap,
            "public_ids_aligned": split_ids_match,
        },
    )
    add_check(
        checks,
        "external_test_only",
        bool((metadata.strat_fold.astype(int) == 10).all())
        and bool((metadata.recommended_split.astype(str) == "external_test").all()),
        metadata.recommended_split.value_counts(dropna=False).to_dict(),
    )
    add_check(
        checks,
        "no_recording_dates",
        bool(metadata.recording_date.isna().all() | metadata.recording_date.astype(str).str.strip().eq("" ).all()),
        int(metadata.recording_date.notna().sum()),
    )
    add_check(
        checks,
        "honest_source_provenance",
        bool((metadata.source_representation == "pdf_vector_reconstructed_lead_segments").all())
        and not bool(metadata.is_original_digital_waveform.astype(bool).any())
        and not bool(metadata.is_fully_synchronous_12lead_10s.astype(bool).any()),
        metadata.source_template.value_counts().to_dict(),
    )

    used_codes: set[str] = set()
    invalid_code_cells: list[int] = []
    for row_number, value in enumerate(metadata.scp_codes, start=1):
        try:
            parsed = ast.literal_eval(value)
            if not isinstance(parsed, dict) or any(float(score) != 100.0 for score in parsed.values()):
                invalid_code_cells.append(row_number)
            used_codes.update(parsed.keys())
        except Exception:
            invalid_code_cells.append(row_number)
    missing_codes = sorted(used_codes.difference(statements.index.astype(str)))
    add_check(
        checks,
        "scp_codes_parse_and_resolve",
        not invalid_code_cells and not missing_codes,
        {"used_codes": len(used_codes), "invalid_rows": invalid_code_cells[:20], "missing": missing_codes},
    )

    records_lines = [line.strip() for line in (public_root / "RECORDS").read_text().splitlines() if line.strip()]
    missing_record_paths = [line for line in records_lines if not (public_root / f"{line}.hea").exists()]
    add_check(
        checks,
        "records_manifest",
        len(records_lines) == 2 * n_records and not missing_record_paths,
        {"entries": len(records_lines), "missing_headers": missing_record_paths[:20]},
    )

    full_data = np.load(public_root / "trace_compat" / "full_data.npy", mmap_mode="r")
    valid_mask = np.load(public_root / "trace_compat" / "valid_mask.npy", mmap_mode="r")
    add_check(
        checks,
        "trace_shapes",
        full_data.shape == (n_records, 12, 2500) and valid_mask.shape == (n_records, 12, 2500),
        {"full_data": list(full_data.shape), "valid_mask": list(valid_mask.shape)},
    )
    finite = bool(np.isfinite(full_data).all())
    max_mean = float(np.max(np.abs(np.mean(full_data, axis=2))))
    max_std_error = float(np.max(np.abs(np.std(full_data, axis=2) - 1.0)))
    add_check(
        checks,
        "trace_finite_and_standardized",
        finite and max_mean < 1e-4 and max_std_error < 1e-3,
        {"finite": finite, "max_abs_lead_mean": max_mean, "max_abs_lead_std_error": max_std_error},
    )

    wfdb_failures: list[dict[str, Any]] = []
    mask_failures: list[dict[str, Any]] = []
    lead_failures: list[dict[str, Any]] = []
    for _, row in metadata.iterrows():
        ecg_id = int(row.ecg_id)
        bucket = f"{(ecg_id // 1000) * 1000:05d}"
        stem = f"{ecg_id:05d}"
        mask_file = public_root / "validity_masks" / bucket / f"{stem}_mask.npz"
        with np.load(mask_file, allow_pickle=False) as masks:
            mask500 = masks["mask500"]
            mask100 = masks["mask100"]
        for rate, suffix, expected_shape, mask in (
            (500, "hr", (5000, 12), mask500),
            (100, "lr", (1000, 12), mask100),
        ):
            record_path = public_root / f"records{rate}" / bucket / f"{stem}_{suffix}"
            try:
                signal, fields = wfdb.rdsamp(str(record_path))
            except Exception as exc:
                wfdb_failures.append({"ecg_id": ecg_id, "rate": rate, "error": str(exc)})
                continue
            if signal.shape != expected_shape or int(fields["fs"]) != rate:
                wfdb_failures.append(
                    {"ecg_id": ecg_id, "rate": rate, "shape": list(signal.shape), "fs": fields["fs"]}
                )
            if tuple(fields["sig_name"]) != LEADS:
                lead_failures.append({"ecg_id": ecg_id, "rate": rate, "leads": fields["sig_name"]})
            if not np.array_equal(np.isfinite(signal), mask):
                mask_failures.append({"ecg_id": ecg_id, "rate": rate})
    add_check(checks, "wfdb_roundtrip_all_records", not wfdb_failures, wfdb_failures[:20])
    add_check(checks, "wfdb_lead_order", not lead_failures, lead_failures[:20])
    add_check(checks, "wfdb_missing_matches_masks", not mask_failures, mask_failures[:20])

    qc = pd.read_csv(public_root / "quality_control" / "record_qc.csv")
    path_count_ok = bool((qc.standard_path_count.astype(int) == 12).all())
    correlation_failures: list[dict[str, Any]] = []
    for _, row in qc.iterrows():
        correlations = json.loads(row.overlap_correlation)
        for lead, correlation in correlations.items():
            if correlation is None or float(correlation) < 0.98:
                correlation_failures.append(
                    {"ecg_id": int(row.ecg_id), "lead": lead, "correlation": correlation}
                )
    add_check(
        checks,
        "waveform_path_counts_and_overlap",
        path_count_ok and not correlation_failures,
        {"path_count_ok": path_count_ok, "overlap_failures": correlation_failures[:20]},
    )

    pdf_files = [str(path.relative_to(public_root)) for path in public_root.rglob("*.pdf")]
    add_check(checks, "no_source_pdfs", not pdf_files, pdf_files[:20])

    public_text: list[tuple[str, str]] = []
    for path in text_files(public_root):
        try:
            public_text.append((str(path.relative_to(public_root)), path.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            continue
    joined_text = "\n".join(text for _, text in public_text)
    leaked_ids = sorted(
        value
        for value in set(mapping.hospital_patient_id.astype(str))
        if value and len(value) >= 6 and value in joined_text
    )
    leaked_basenames = sorted(
        value
        for value in {Path(item).name for item in mapping.source_relative_path.astype(str)}
        if value and value in joined_text
    )
    add_check(
        checks,
        "no_source_ids_or_filenames_in_public_text",
        not leaked_ids and not leaked_basenames,
        {"hospital_ids": leaked_ids[:20], "source_basenames": leaked_basenames[:20]},
    )
    report_text = "\n".join(metadata.report.fillna("").astype(str))
    forbidden = re.findall(
        r"姓名|患者编号|病历号|登记号|住院号|门诊号|医师|医生|"
        r"(?<!\d)(?:19|20)\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2}(?:日)?(?!\d)|"
        r"(?<!\d)\d{7,}(?!\d)",
        report_text,
    )
    add_check(checks, "diagnosis_text_phi_pattern_scan", not forbidden, forbidden[:20])

    required_governance = (
        "README.md",
        "DATA_DICTIONARY.md",
        "ETHICS_AND_GOVERNANCE_REQUIRED.md",
        "LICENSE_PENDING.txt",
        "CITATION.cff",
    )
    missing_governance = [name for name in required_governance if not (public_root / name).exists()]
    add_check(checks, "release_documentation_present", not missing_governance, missing_governance)

    failures = [check for check in checks if not check["passed"]]
    report = {
        "status": "PASS" if not failures else "FAIL",
        "record_count": n_records,
        "patient_count": int(metadata.patient_id.nunique()),
        "checks": checks,
        "failure_count": len(failures),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    sys.exit(main())
