#!/usr/bin/env python3
"""Build a provenance-preserving derived augmentation pack for Anzhen ECG.

This script creates label-preserving training augmentations only. It does not
create new diagnoses, new patients, or clinical validation records. Source
records are restricted to the patient-level training partition supplied by a
locked split manifest. Validation and test records are never read as waveform
parents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_REAL_ROOT = REPO_ROOT / "data/anzhen_external"
DEFAULT_SPLIT_MANIFEST = Path(
    REPO_ROOT / "results/anzhen_report_control/split_50_50_seed42/split_manifest.csv"
)

TARGET_COLUMNS = {
    "form_NDT": "target_form_NDT",
    "form_STD_": "target_form_STD_",
    "form_NST_": "target_form_NST_",
    "form_QWAVE": "target_form_QWAVE",
    "form_LOWT": "target_form_LOWT",
    "rhythm_SR": "target_rhythm_SR",
    "rhythm_AFIB": "target_rhythm_AFIB",
    "rhythm_STACH": "target_rhythm_STACH",
    "rhythm_SARRH": "target_rhythm_SARRH",
    "rhythm_SBRAD": "target_rhythm_SBRAD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate conservative, label-preserving derived ECG variants "
            "from real Anzhen training records only"
        )
    )
    parser.add_argument("--real-root", type=Path, default=DEFAULT_REAL_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination package; must be outside this source-code directory.",
    )
    parser.add_argument("--target-min-positive", type=int, default=100)
    parser.add_argument("--max-variants-per-patient", type=int, default=5)
    parser.add_argument("--max-synthetic-records", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def choose_parents(
    train: pd.DataFrame,
    target_min_positive: int,
    max_variants_per_patient: int,
    max_synthetic_records: int,
    rng: np.random.Generator,
) -> tuple[list[int], dict[str, int], dict[str, int], dict[str, int]]:
    columns = list(TARGET_COLUMNS.values())
    labels = list(TARGET_COLUMNS)
    matrix = train[columns].to_numpy(dtype=np.int8)
    real_counts = dict(zip(labels, matrix.sum(axis=0).astype(int).tolist()))
    targets = {
        label: max(real_counts[label], int(target_min_positive)) for label in labels
    }
    effective = dict(real_counts)
    patient_ids = train["patient_id"].astype(str).to_numpy()
    patient_usage: dict[str, int] = {patient: 0 for patient in np.unique(patient_ids)}
    selected: list[int] = []

    while len(selected) < max_synthetic_records:
        deficit = {
            label: max(0, targets[label] - effective[label]) for label in labels
        }
        active_deficit_indices = [index for index, label in enumerate(labels) if deficit[label] > 0]
        if not active_deficit_indices:
            break

        scores = np.zeros(len(train), dtype=np.float64)
        for label_index in active_deficit_indices:
            label = labels[label_index]
            # Relative deficit gives the rarest unresolved labels priority.
            scores += matrix[:, label_index] * (
                deficit[label] / max(float(targets[label]), 1.0)
            )
        capacity = np.asarray(
            [patient_usage[patient] < max_variants_per_patient for patient in patient_ids],
            dtype=bool,
        )
        eligible = np.flatnonzero((scores > 0) & capacity)
        if eligible.size == 0:
            break

        adjusted = scores[eligible] / np.asarray(
            [1.0 + patient_usage[patient_ids[index]] for index in eligible],
            dtype=np.float64,
        )
        probabilities = adjusted / adjusted.sum()
        selected_position = int(rng.choice(eligible, p=probabilities))
        selected.append(selected_position)
        patient_usage[patient_ids[selected_position]] += 1
        for label_index, label in enumerate(labels):
            if matrix[selected_position, label_index]:
                effective[label] += 1

    return selected, real_counts, targets, effective


def augment_signal(
    signal: np.ndarray,
    valid_mask: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    if signal.shape != (12, 2500) or valid_mask.shape != signal.shape:
        raise ValueError(
            f"Expected signal/mask shaped (12, 2500), got {signal.shape}/{valid_mask.shape}"
        )

    shift_samples = int(rng.integers(-25, 26))
    shifted = np.roll(signal.astype(np.float64), shift_samples, axis=1)
    shifted_mask = np.roll(valid_mask.astype(bool), shift_samples, axis=1)

    sampling_rate = 250.0
    time = np.arange(signal.shape[1], dtype=np.float64) / sampling_rate
    lead_gain = rng.uniform(0.97, 1.03, size=(12, 1))
    baseline_frequency_hz = rng.uniform(0.10, 0.35, size=(12, 1))
    baseline_phase_rad = rng.uniform(0.0, 2.0 * math.pi, size=(12, 1))
    baseline_amplitude_sd = rng.uniform(0.005, 0.025, size=(12, 1))
    baseline = baseline_amplitude_sd * np.sin(
        2.0 * math.pi * baseline_frequency_hz * time.reshape(1, -1)
        + baseline_phase_rad
    )
    white_noise_sd = rng.uniform(0.003, 0.012, size=(12, 1))
    noise = rng.normal(0.0, 1.0, size=signal.shape) * white_noise_sd

    augmented = shifted * lead_gain + baseline + noise
    per_lead_mean = augmented.mean(axis=1, keepdims=True)
    per_lead_std = augmented.std(axis=1, keepdims=True)
    if bool(np.any(~np.isfinite(per_lead_std))) or bool(np.any(per_lead_std < 1e-8)):
        raise RuntimeError("Synthetic transformation produced a constant or non-finite lead")
    augmented = (augmented - per_lead_mean) / (per_lead_std + 1e-6)
    augmented = augmented.astype(np.float32)
    if not np.isfinite(augmented).all():
        raise RuntimeError("Synthetic transformation produced NaN or infinite values")

    # Similarity is calculated after undoing the circular shift. This is a
    # transformation QC value, not evidence of clinical validity.
    aligned = np.roll(augmented, -shift_samples, axis=1)
    correlations = []
    for lead_index in range(12):
        correlations.append(
            float(np.corrcoef(signal[lead_index], aligned[lead_index])[0, 1])
        )
    parameters = {
        "time_shift_samples_at_250hz": shift_samples,
        "lead_gain": lead_gain.reshape(-1).tolist(),
        "baseline_frequency_hz": baseline_frequency_hz.reshape(-1).tolist(),
        "baseline_amplitude_sd": baseline_amplitude_sd.reshape(-1).tolist(),
        "white_noise_sd": white_noise_sd.reshape(-1).tolist(),
        "post_transform_per_lead_standardization": True,
    }
    qc = {
        "mean_max_abs": float(np.max(np.abs(augmented.mean(axis=1)))),
        "std_max_abs_error": float(np.max(np.abs(augmented.std(axis=1) - 1.0))),
        "max_abs": float(np.max(np.abs(augmented))),
        "parent_similarity_aligned_pearson_mean": float(np.mean(correlations)),
        "parent_similarity_aligned_pearson_min": float(np.min(correlations)),
        "valid_fraction": float(shifted_mask.mean()),
    }
    return augmented, shifted_mask, parameters, qc


def ensure_clean_output(output_dir: Path, overwrite: bool) -> None:
    generated = [
        "synthetic_signals.npy",
        "synthetic_valid_mask.npy",
        "synthetic_metadata.csv",
        "quality_control.csv",
        "label_distribution.csv",
        "parent_usage.csv",
        "protocol.json",
        "MANIFEST.csv",
        "SHA256SUMS.txt",
    ]
    existing = [output_dir / name for name in generated if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Generated outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def build_manifest(output_dir: Path) -> None:
    excluded = {"MANIFEST.csv", "SHA256SUMS.txt"}
    paths = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name not in excluded
    )
    rows = [
        {
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    pd.DataFrame(rows).to_csv(output_dir / "MANIFEST.csv", index=False)
    (output_dir / "SHA256SUMS.txt").write_text(
        "".join(f"{row['sha256']}  {row['file']}\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.target_min_positive <= 0:
        raise ValueError("--target-min-positive must be positive")
    if args.max_variants_per_patient <= 0:
        raise ValueError("--max-variants-per-patient must be positive")
    ensure_clean_output(args.output_dir, args.overwrite)

    signal_path = args.real_root / "trace_compat/full_data.npy"
    mask_path = args.real_root / "trace_compat/valid_mask.npy"
    for required in (signal_path, mask_path, args.split_manifest):
        if not required.exists():
            raise FileNotFoundError(required)

    signals = np.load(signal_path, mmap_mode="r")
    masks = np.load(mask_path, mmap_mode="r")
    split = pd.read_csv(args.split_manifest)
    if signals.shape != masks.shape or tuple(signals.shape[1:]) != (12, 2500):
        raise ValueError(f"Unexpected source shapes: signals={signals.shape}, masks={masks.shape}")
    if int(split["row_index"].max()) >= signals.shape[0]:
        raise ValueError("Split manifest row_index exceeds source waveform array")
    required_columns = {
        "row_index",
        "ecg_id",
        "record_uid",
        "patient_id",
        "split",
        *TARGET_COLUMNS.values(),
    }
    missing_columns = sorted(required_columns - set(split.columns))
    if missing_columns:
        raise ValueError(f"Split manifest lacks required columns: {missing_columns}")

    train = split.loc[split["split"].astype(str).eq("train")].copy().reset_index(drop=True)
    if train.empty:
        raise ValueError("Split manifest contains no training records")
    rng = np.random.default_rng(args.seed)
    selected, real_counts, targets, effective_counts = choose_parents(
        train,
        args.target_min_positive,
        args.max_variants_per_patient,
        args.max_synthetic_records,
        rng,
    )
    if not selected:
        raise RuntimeError("No synthetic records were selected")

    synthetic_signals = np.lib.format.open_memmap(
        args.output_dir / "synthetic_signals.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(selected), 12, 2500),
    )
    synthetic_masks = np.lib.format.open_memmap(
        args.output_dir / "synthetic_valid_mask.npy",
        mode="w+",
        dtype=bool,
        shape=(len(selected), 12, 2500),
    )

    metadata_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    parent_usage: dict[str, int] = {}
    target_columns_all = [column for column in split.columns if column.startswith("target_")]
    label_columns_native = [column for column in split.columns if column.endswith("_labels_native")]
    for synthetic_index, selected_position in enumerate(selected):
        parent = train.iloc[selected_position]
        source_row_index = int(parent["row_index"])
        sample_seed = int(rng.integers(0, 2**32 - 1))
        sample_rng = np.random.default_rng(sample_seed)
        augmented, augmented_mask, parameters, qc = augment_signal(
            np.asarray(signals[source_row_index], dtype=np.float32),
            np.asarray(masks[source_row_index], dtype=bool),
            sample_rng,
        )
        synthetic_signals[synthetic_index] = augmented
        synthetic_masks[synthetic_index] = augmented_mask

        synthetic_id = f"AZSYNV01_{synthetic_index + 1:06d}"
        parent_patient = str(parent["patient_id"])
        parent_usage[parent_patient] = parent_usage.get(parent_patient, 0) + 1
        positive_labels = [
            label
            for label, column in TARGET_COLUMNS.items()
            if int(parent[column]) == 1
        ]
        row: dict[str, Any] = {
            "synthetic_index": synthetic_index,
            "synthetic_id": synthetic_id,
            "sample_origin": "synthetic",
            "independent_patient_contribution": 0,
            "intended_use": "training_augmentation_only",
            "clinical_validation_eligible": False,
            "label_origin": "copied_from_real_training_parent",
            "generation_family": "label_preserving_signal_perturbation",
            "source_split": str(parent["split"]),
            "parent_row_index": source_row_index,
            "parent_ecg_id": parent["ecg_id"],
            "parent_record_uid": parent["record_uid"],
            "parent_patient_id": parent_patient,
            "patient_id": parent_patient,
            "source_template": parent.get("source_template", ""),
            "layout_subtype": parent.get("layout_subtype", ""),
            "sample_seed": sample_seed,
            "positive_ptb_aligned_control_labels": json.dumps(
                positive_labels, ensure_ascii=False
            ),
            "transformation_parameters": json.dumps(parameters, ensure_ascii=False),
        }
        for column in label_columns_native + target_columns_all:
            row[column] = parent[column]
        metadata_rows.append(row)
        qc_rows.append(
            {
                "synthetic_index": synthetic_index,
                "synthetic_id": synthetic_id,
                "parent_row_index": source_row_index,
                "parent_patient_id": parent_patient,
                **qc,
                "automated_qc_pass": bool(
                    qc["mean_max_abs"] < 1e-4
                    and qc["std_max_abs_error"] < 1e-4
                    and qc["parent_similarity_aligned_pearson_min"] >= 0.98
                    and qc["max_abs"] < 12.0
                ),
            }
        )

    synthetic_signals.flush()
    synthetic_masks.flush()
    metadata = pd.DataFrame(metadata_rows)
    qc_frame = pd.DataFrame(qc_rows)
    metadata.to_csv(args.output_dir / "synthetic_metadata.csv", index=False)
    qc_frame.to_csv(args.output_dir / "quality_control.csv", index=False)
    pd.DataFrame(
        [
            {
                "label": label,
                "target_column": TARGET_COLUMNS[label],
                "real_train_positive": real_counts[label],
                "requested_effective_minimum": targets[label],
                "synthetic_positive": int(metadata[TARGET_COLUMNS[label]].sum()),
                "effective_real_plus_synthetic_positive": effective_counts[label],
                "target_reached": effective_counts[label] >= targets[label],
            }
            for label in TARGET_COLUMNS
        ]
    ).to_csv(args.output_dir / "label_distribution.csv", index=False)
    pd.DataFrame(
        [
            {"parent_patient_id": patient, "synthetic_variants": count}
            for patient, count in sorted(parent_usage.items())
        ]
    ).to_csv(args.output_dir / "parent_usage.csv", index=False)

    source_splits = set(metadata["source_split"].astype(str))
    if source_splits != {"train"}:
        raise RuntimeError(f"Non-training parents detected: {sorted(source_splits)}")
    if int(metadata.groupby("parent_patient_id").size().max()) > args.max_variants_per_patient:
        raise RuntimeError("Per-patient variant cap was exceeded")
    if not bool(qc_frame["automated_qc_pass"].all()):
        failures = int((~qc_frame["automated_qc_pass"]).sum())
        raise RuntimeError(f"Automated QC failed for {failures} synthetic records")

    protocol = {
        "version": "0.1.0",
        "status": "derived_training_augmentation_pack",
        "purpose": "label-preserving augmentation of the Anzhen report-control training partition",
        "not_claimed_as": [
            "new patients",
            "new clinical ECG acquisitions",
            "guideline-derived disease generation",
            "validation or test evidence",
        ],
        "source": {
            "real_dataset_root": args.real_root.name,
            "signals": "trace_compat/full_data.npy",
            "signals_sha256": sha256_file(signal_path),
            "valid_mask": "trace_compat/valid_mask.npy",
            "valid_mask_sha256": sha256_file(mask_path),
            "split_manifest": args.split_manifest.name,
            "split_manifest_sha256": sha256_file(args.split_manifest),
            "eligible_parent_split": "train only",
            "real_train_records": int(len(train)),
            "real_train_patients": int(train["patient_id"].nunique()),
        },
        "generation": {
            "seed": args.seed,
            "target_min_positive_per_ptb_aligned_control_label": args.target_min_positive,
            "max_variants_per_patient": args.max_variants_per_patient,
            "max_synthetic_records": args.max_synthetic_records,
            "generated_records": int(len(metadata)),
            "generated_independent_patients": 0,
            "transformations": [
                "circular time shift of at most 25 samples at 250 Hz",
                "lead gain in [0.97, 1.03]",
                "low-amplitude baseline sinusoid",
                "low-amplitude white noise",
                "per-lead re-standardization",
            ],
            "label_policy": "all labels copied unchanged from the real training parent",
            "disease_creation": "none",
        },
        "label_distribution": {
            label: {
                "real_train_positive": real_counts[label],
                "target": targets[label],
                "synthetic_positive": int(metadata[TARGET_COLUMNS[label]].sum()),
                "effective_real_plus_synthetic_positive": effective_counts[label],
            }
            for label in TARGET_COLUMNS
        },
        "quality_control": {
            "automated_qc_passed": int(qc_frame["automated_qc_pass"].sum()),
            "automated_qc_failed": int((~qc_frame["automated_qc_pass"]).sum()),
            "minimum_parent_similarity_threshold": 0.98,
        },
        "governance": {
            "paper_cohort_count_policy": "synthetic records must not be counted as patients or real ECG acquisitions",
            "evaluation_policy": "synthetic records must not enter validation, test, bootstrap confidence intervals, or clinical performance claims",
            "release_policy": "publish separately from the real dataset with explicit origin and provenance fields",
        },
    }
    write_json(args.output_dir / "protocol.json", protocol)
    build_manifest(args.output_dir)
    print(json.dumps(json_safe(protocol), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
