#!/usr/bin/env python3
"""Patient-level Anzhen adaptation of the TRACE report-control head.

The protocol reuses the locked 50:50 diagnostic patient assignment.  Patients
without a five-superclass diagnosis label are assigned afterwards so that the
complete control-label cohort remains 40% train, 10% validation, and 50% locked
test.  The PTB-XL control vocabulary is intentionally preserved: this keeps the
adapted checkpoint compatible with the already-trained TRACE report generator.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from evaluate_anzhen_external import (
    DEFAULT_DATA_ROOT,
    json_safe,
    load_cohort,
    sha256_file,
    sigmoid,
    write_json,
)
from finetune_anzhen_diagnosis import iterative_multilabel_assignment
from utils.checkpoint_compat import remap_legacy_foundation_encoder_keys
from utils.downstream_trace import slice_signal_window
from utils.report_controls import (
    CONTROL_GROUPS,
    ReportControlClassifier,
    add_ptbxl_diagnostic_superclass_column,
    build_control_targets,
    compute_ptbxl_label_aggregations,
    parse_control_dict,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_CHECKPOINT = REPO_ROOT / "ckpt/ptbxl/report_control/best.pt"
DEFAULT_TOKENIZER_CHECKPOINT = (
    REPO_ROOT
    / "ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth"
)
DEFAULT_LOCKED_SPLIT = None
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "results/anzhen_report_control/split_50_50_seed42"
)
CONTROL_COLUMNS = {
    "diagnostic": "diagnostic_superclass",
    "form": "form",
    "rhythm": "rhythm",
}
GROUP_LOSS_DEFAULTS = {"diagnostic": 0.25, "form": 1.0, "rhythm": 1.0}


@dataclass(frozen=True)
class ControlCohort:
    signals: np.ndarray
    metadata: pd.DataFrame
    eligible_rows: np.ndarray
    targets: dict[str, np.ndarray]
    source_vocab: dict[str, list[str]]
    active_labels: dict[str, list[str]]
    inactive_labels: dict[str, list[str]]
    native_label_counts: dict[str, dict[str, int]]


@dataclass(frozen=True)
class SplitArtifacts:
    manifest: pd.DataFrame
    indices: dict[str, np.ndarray]
    summary: dict[str, Any]


@dataclass(frozen=True)
class SyntheticAugmentation:
    signals: np.ndarray
    targets: dict[str, np.ndarray]
    metadata: pd.DataFrame
    provenance: dict[str, Any]


class AnzhenControlDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        targets: Mapping[str, np.ndarray],
        indices: np.ndarray,
    ) -> None:
        self.signals = signals
        self.targets = targets
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int):
        row = int(self.indices[index])
        signal = np.array(self.signals[row], dtype=np.float32, copy=True)
        outputs = [torch.from_numpy(signal)]
        for group in CONTROL_GROUPS:
            outputs.append(
                torch.from_numpy(np.array(self.targets[group][row], dtype=np.float32, copy=True))
            )
        outputs.append(row)
        return tuple(outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adapt the PTB-XL TRACE report-control head on Anzhen using a locked 50:50 protocol"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=Path,
        default=DEFAULT_TOKENIZER_CHECKPOINT,
        help="Tokenizer used to reconstruct the report-control model before strict checkpoint loading.",
    )
    parser.add_argument(
        "--locked-split-manifest",
        type=Path,
        default=DEFAULT_LOCKED_SPLIT,
        help=(
            "Frozen patient-level split. Defaults to "
            "DATA_ROOT/trace_compat/paper_report_control_split_seed42.csv."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--derived-root",
        "--synthetic-root",
        dest="synthetic_root",
        type=Path,
        default=None,
        metavar="DERIVED_ROOT",
        help="Optional train-only package of clinician-reviewed label-preserving derived variants.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=512)
    parser.add_argument("--split-label-min-count", type=int, default=5)
    parser.add_argument("--min-active-label-count", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--diagnostic-loss-weight", type=float, default=0.25)
    parser.add_argument("--form-loss-weight", type=float, default=1.0)
    parser.add_argument("--rhythm-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--target-metric",
        choices=("prompt_macro_auroc", "prompt_macro_auprc"),
        default="prompt_macro_auroc",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seq-length", type=int, default=2250)
    parser.add_argument("--latent-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test only")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Smoke-test only")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run training/validation plumbing only and exit before the locked test set is accessed",
    )
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(max_samples=None)
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_seed(worker_id: int) -> None:
    del worker_id
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def source_vocab_from_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, list[str]]:
    vocab = {
        group: [str(label) for label in checkpoint["control_vocab"].get(group, [])]
        for group in CONTROL_GROUPS
    }
    if any(not vocab[group] for group in CONTROL_GROUPS):
        raise ValueError(f"Source checkpoint has an empty control group: {vocab}")
    return vocab


def count_labels(values: Sequence[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        for label in parse_control_dict(value):
            counts[label] = counts.get(label, 0) + 1
    return counts


def load_control_cohort(
    args: argparse.Namespace,
    source_vocab: Mapping[str, Sequence[str]],
) -> ControlCohort:
    base = load_cohort(args)
    statements_path = args.data_root / "anzhen_statements.csv"
    if not statements_path.exists():
        statements_path = args.data_root / "trace_compat/scp_statements.csv"

    metadata = base.metadata.copy()
    metadata["scp_codes"] = metadata["scp_codes"].apply(
        lambda value: value if isinstance(value, Mapping) else ast.literal_eval(str(value))
    )
    metadata = compute_ptbxl_label_aggregations(metadata, str(statements_path))
    metadata = add_ptbxl_diagnostic_superclass_column(metadata, str(statements_path))
    metadata["row_index"] = np.arange(len(metadata), dtype=np.int64)

    native_counts = {
        group: count_labels(metadata[CONTROL_COLUMNS[group]].tolist())
        for group in CONTROL_GROUPS
    }
    eligible = np.zeros(len(metadata), dtype=bool)
    for group in CONTROL_GROUPS:
        eligible |= metadata[CONTROL_COLUMNS[group]].apply(
            lambda value: bool(parse_control_dict(value))
        ).to_numpy(dtype=bool)
    eligible_rows = np.flatnonzero(eligible)
    if eligible_rows.size == 0:
        raise ValueError("No Anzhen rows contain mapped report-control labels")

    targets = build_control_targets(metadata, source_vocab, column_map=CONTROL_COLUMNS)
    active_labels: dict[str, list[str]] = {}
    inactive_labels: dict[str, list[str]] = {}
    for group in CONTROL_GROUPS:
        active_labels[group] = [
            label
            for label in source_vocab[group]
            if native_counts[group].get(label, 0) >= args.min_active_label_count
        ]
        inactive_labels[group] = [
            label for label in source_vocab[group] if label not in active_labels[group]
        ]
        if not active_labels[group]:
            raise ValueError(f"No active labels remain for group {group}")

    return ControlCohort(
        signals=base.signals,
        metadata=metadata,
        eligible_rows=eligible_rows,
        targets=targets,
        source_vocab={group: list(source_vocab[group]) for group in CONTROL_GROUPS},
        active_labels=active_labels,
        inactive_labels=inactive_labels,
        native_label_counts=native_counts,
    )


def load_synthetic_augmentation(
    root: Path,
    source_vocab: Mapping[str, Sequence[str]],
    split: SplitArtifacts,
    seq_length: int,
    latent_ratio: float,
) -> SyntheticAugmentation:
    signal_path = root / "synthetic_signals.npy"
    mask_path = root / "synthetic_valid_mask.npy"
    metadata_path = root / "synthetic_metadata.csv"
    protocol_path = root / "protocol.json"
    for required in (signal_path, mask_path, metadata_path, protocol_path):
        if not required.exists():
            raise FileNotFoundError(f"Synthetic package is missing {required}")

    raw_signals = np.load(signal_path, mmap_mode="r")
    raw_masks = np.load(mask_path, mmap_mode="r")
    metadata = pd.read_csv(metadata_path)
    if raw_signals.shape != raw_masks.shape or tuple(raw_signals.shape[1:]) != (12, 2500):
        raise ValueError(
            f"Unexpected synthetic signal/mask shapes: {raw_signals.shape}/{raw_masks.shape}"
        )
    if len(metadata) != len(raw_signals):
        raise ValueError(
            f"Synthetic metadata/signals length mismatch: {len(metadata)}/{len(raw_signals)}"
        )
    signals = slice_signal_window(
        raw_signals,
        seq_length=seq_length,
        latent_ratio=latent_ratio,
    )
    if not np.isfinite(signals).all():
        raise ValueError("Synthetic signals contain NaN or infinite values")

    required_columns = {
        "synthetic_id",
        "synthetic_index",
        "sample_origin",
        "independent_patient_contribution",
        "intended_use",
        "clinical_validation_eligible",
        "source_split",
        "parent_row_index",
        "parent_patient_id",
        "patient_id",
    }
    target_columns = {
        f"target_{group}_{label}"
        for group in CONTROL_GROUPS
        for label in source_vocab[group]
    }
    missing = sorted((required_columns | target_columns) - set(metadata.columns))
    if missing:
        raise ValueError(f"Synthetic metadata lacks required columns: {missing}")
    if set(metadata["sample_origin"].astype(str)) != {"synthetic"}:
        raise ValueError("Synthetic package contains a non-synthetic sample_origin")
    if set(metadata["source_split"].astype(str)) != {"train"}:
        raise ValueError("Synthetic parents are not restricted to the real training split")
    if set(metadata["intended_use"].astype(str)) != {"training_augmentation_only"}:
        raise ValueError("Synthetic package contains a non-training intended_use")
    if bool((metadata["independent_patient_contribution"].astype(int) != 0).any()):
        raise ValueError("Synthetic records must contribute zero independent patients")
    if bool(metadata["clinical_validation_eligible"].astype(str).str.lower().isin({"true", "1"}).any()):
        raise ValueError("Synthetic records must not be eligible for clinical validation")
    if not metadata["parent_patient_id"].astype(str).equals(metadata["patient_id"].astype(str)):
        raise ValueError("Synthetic patient_id must equal parent_patient_id for leakage-safe grouping")
    expected_index = np.arange(len(metadata), dtype=np.int64)
    if not np.array_equal(metadata["synthetic_index"].to_numpy(dtype=np.int64), expected_index):
        raise ValueError("Synthetic indices must be contiguous and aligned with the signal array")

    real_manifest = split.manifest.set_index("row_index", drop=False)
    parent_rows = metadata["parent_row_index"].to_numpy(dtype=np.int64)
    missing_rows = sorted(set(parent_rows) - set(real_manifest.index.astype(int)))
    if missing_rows:
        raise ValueError(f"Synthetic parents are absent from the real manifest: {missing_rows[:5]}")
    parent_manifest = real_manifest.loc[parent_rows]
    if set(parent_manifest["split"].astype(str)) != {"train"}:
        raise ValueError("At least one synthetic parent belongs to validation or test")
    if not np.array_equal(
        parent_manifest["patient_id"].astype(str).to_numpy(),
        metadata["parent_patient_id"].astype(str).to_numpy(),
    ):
        raise ValueError("Synthetic parent patient identifiers do not match the real manifest")

    targets: dict[str, np.ndarray] = {}
    for group in CONTROL_GROUPS:
        columns = [f"target_{group}_{label}" for label in source_vocab[group]]
        matrix = metadata[columns].to_numpy(dtype=np.float32)
        if bool(np.any((matrix != 0) & (matrix != 1))):
            raise ValueError(f"Synthetic {group} targets are not binary")
        parent_matrix = parent_manifest[columns].to_numpy(dtype=np.float32)
        if not np.array_equal(matrix, parent_matrix):
            raise ValueError(f"Synthetic {group} labels differ from their real parents")
        targets[group] = matrix

    split_sets = {
        name: set(split.manifest.loc[split.manifest["split"].eq(name), "patient_id"].astype(str))
        for name in ("train", "valid", "test")
    }
    parent_patients = set(metadata["parent_patient_id"].astype(str))
    leakage = {
        "validation_parent_overlap": len(parent_patients & split_sets["valid"]),
        "test_parent_overlap": len(parent_patients & split_sets["test"]),
    }
    if any(leakage.values()):
        raise RuntimeError(f"Synthetic parent leakage detected: {leakage}")

    provenance = {
        "root": str(root),
        "signals": str(signal_path),
        "signals_sha256": sha256_file(signal_path),
        "valid_mask": str(mask_path),
        "valid_mask_sha256": sha256_file(mask_path),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "records": int(len(metadata)),
        "stored_signal_shape": list(raw_signals.shape),
        "training_signal_shape": list(signals.shape),
        "training_window_rule": {
            "seq_length": int(seq_length),
            "latent_ratio": float(latent_ratio),
            "implementation": "same slice_signal_window function used for real Anzhen records",
        },
        "parent_patients": int(len(parent_patients)),
        "new_independent_patients": 0,
        "source_split": "train only",
        "validation_and_test_records": 0,
        "validation_parent_overlap": 0,
        "test_parent_overlap": 0,
    }
    return SyntheticAugmentation(
        signals=signals,
        targets=targets,
        metadata=metadata,
        provenance=provenance,
    )


def protocol_split_sizes(n_patients: int) -> dict[str, int]:
    """Exact two-stage 50:50 split, then 80:20 within development."""
    test = (n_patients + 1) // 2
    development = n_patients - test
    train = int(math.floor(0.8 * development + 0.5))
    valid = development - train
    return {"train": train, "valid": valid, "test": test}


def global_assignment_score(
    assignment: np.ndarray,
    features: np.ndarray,
    label_feature_count: int,
    split_sizes: Mapping[str, int],
    patient_record_counts: np.ndarray,
) -> tuple[float, float, float, float]:
    split_names = ("train", "valid", "test")
    capacities = np.asarray([split_sizes[name] for name in split_names], dtype=float)
    ratios = capacities / capacities.sum()
    actual = np.stack(
        [features[assignment == index].sum(axis=0) for index in range(len(split_names))]
    )
    expected = ratios[:, None] * features.sum(axis=0, keepdims=True)
    scaled_error = np.abs(actual - expected) / np.maximum(expected, 1.0)
    label_error = scaled_error[:, :label_feature_count]
    layout_error = scaled_error[:, label_feature_count:]
    record_actual = np.asarray(
        [patient_record_counts[assignment == index].sum() for index in range(len(split_names))],
        dtype=float,
    )
    record_expected = ratios * patient_record_counts.sum()
    record_error = np.abs(record_actual - record_expected) / np.maximum(record_expected, 1.0)
    return (
        float(label_error.max()),
        float(np.mean(label_error**2)),
        float(np.mean(layout_error**2)) if layout_error.size else 0.0,
        float(np.mean(record_error**2)),
    )


def build_patient_split(
    cohort: ControlCohort,
    locked_manifest_path: Path,
    seed: int,
    trials: int,
    split_label_min_count: int,
) -> SplitArtifacts:
    metadata = cohort.metadata.iloc[cohort.eligible_rows].copy()
    metadata["patient_id"] = metadata["patient_id"].astype(str)
    locked = pd.read_csv(locked_manifest_path)
    frozen_report_control_manifest = "paper_partition" in locked.columns
    if frozen_report_control_manifest:
        required_frozen = {
            "row_index",
            "record_uid",
            "patient_id",
            "paper_partition",
            "eligible_for_report_control",
        }
        if not required_frozen.issubset(locked.columns):
            raise ValueError(
                "Frozen report-control manifest lacks columns "
                f"{sorted(required_frozen - set(locked.columns))}"
            )
        eligible_flag = locked["eligible_for_report_control"]
        if eligible_flag.dtype != bool:
            eligible_flag = eligible_flag.astype(str).str.strip().str.lower().eq("true")
        excluded = locked["paper_partition"].astype(str).eq("excluded")
        if not bool((eligible_flag == ~excluded).all()):
            raise ValueError(
                "Frozen manifest eligibility flags do not agree with paper_partition"
            )
        frozen = locked.loc[eligible_flag].copy()
        frozen = frozen.rename(columns={"paper_partition": "split"})
        expected_rows = metadata.index.to_numpy(dtype=np.int64)
        observed_rows = frozen["row_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(np.sort(observed_rows), np.sort(expected_rows)):
            raise ValueError(
                "Frozen report-control manifest rows do not exactly match the eligible cohort"
            )
        aligned = frozen.set_index("row_index").loc[expected_rows]
        if not np.array_equal(
            aligned["record_uid"].astype(str).to_numpy(),
            metadata["record_uid"].astype(str).to_numpy(),
        ) or not np.array_equal(
            aligned["patient_id"].astype(str).to_numpy(),
            metadata["patient_id"].astype(str).to_numpy(),
        ):
            raise ValueError("Frozen report-control manifest public IDs are not row-aligned")
        locked = frozen
        locked_source_label = "locked_report_control_split"
    else:
        locked_source_label = "locked_diagnosis_50_50"
    required = {"patient_id", "split", "row_index"}
    if not required.issubset(locked.columns):
        raise ValueError(f"Locked split manifest lacks columns {sorted(required - set(locked.columns))}")
    locked["patient_id"] = locked["patient_id"].astype(str)
    invalid_splits = sorted(set(locked["split"].astype(str)) - {"train", "valid", "test"})
    if invalid_splits:
        raise ValueError(f"Invalid split names in locked manifest: {invalid_splits}")
    conflicts = locked.groupby("patient_id")["split"].nunique()
    if bool((conflicts > 1).any()):
        raise ValueError("The locked manifest assigns at least one patient to multiple splits")
    locked_map = locked.drop_duplicates("patient_id").set_index("patient_id")["split"].to_dict()

    eligible_patients = set(metadata["patient_id"])
    missing_locked = sorted(set(locked_map) - eligible_patients)
    if missing_locked:
        raise ValueError(f"Locked diagnostic patients missing from control cohort: {missing_locked[:5]}")

    feature_values: dict[str, np.ndarray] = {}
    label_feature_names: list[str] = []
    for group in CONTROL_GROUPS:
        counts = cohort.native_label_counts[group]
        labels = sorted(
            label for label, count in counts.items() if count >= split_label_min_count
        )
        dictionaries = metadata[CONTROL_COLUMNS[group]].apply(parse_control_dict)
        for label in labels:
            name = f"{group}::{label}"
            label_feature_names.append(name)
            feature_values[name] = dictionaries.apply(lambda item, key=label: int(key in item)).to_numpy()

    normalized_layout = metadata["layout_subtype"].astype(str).replace(
        {"PSCRIPT_BUNDLE": "PSCRIPT"}
    )
    layout_feature_names = [
        f"layout::{name}" for name in sorted(normalized_layout.unique().tolist())
    ]
    for name in layout_feature_names:
        layout = name.split("::", 1)[1]
        feature_values[name] = normalized_layout.eq(layout).astype(np.int8).to_numpy()

    row_features = pd.DataFrame(feature_values, index=metadata.index)
    row_features["patient_id"] = metadata["patient_id"].to_numpy()
    row_features["record_count"] = 1
    feature_names = label_feature_names + layout_feature_names
    patient_frame = row_features.groupby("patient_id", sort=True)[feature_names].max()
    patient_record_counts = (
        row_features.groupby("patient_id", sort=True)["record_count"].sum().to_numpy(dtype=np.int64)
    )
    patient_ids = patient_frame.index.to_numpy(dtype=str)
    features = patient_frame.to_numpy(dtype=np.int8)
    patient_to_position = {patient: index for index, patient in enumerate(patient_ids)}

    split_names = ("train", "valid", "test")
    split_to_index = {name: index for index, name in enumerate(split_names)}
    split_sizes = protocol_split_sizes(len(patient_ids))
    locked_counts = {
        name: int(sum(value == name for value in locked_map.values())) for name in split_names
    }
    remaining_sizes = {
        name: split_sizes[name] - locked_counts[name] for name in split_names
    }
    if any(value < 0 for value in remaining_sizes.values()):
        raise ValueError(
            f"Locked patient counts exceed target 50:50 capacities: locked={locked_counts}, target={split_sizes}"
        )

    new_patients = [patient for patient in patient_ids if patient not in locked_map]
    if sum(remaining_sizes.values()) != len(new_patients):
        raise RuntimeError("Remaining split capacities do not match the number of unassigned patients")
    new_positions = np.asarray([patient_to_position[patient] for patient in new_patients], dtype=np.int64)
    new_features = features[new_positions]
    if trials <= 0:
        raise ValueError("--split-trials must be positive")

    best_assignment: np.ndarray | None = None
    best_score: tuple[float, float, float, float] | None = None
    best_trial = -1
    trial_indices = range(trials) if new_patients else range(1)
    for trial in trial_indices:
        if new_patients:
            rng = np.random.default_rng(seed + 104729 * trial)
            new_assignment = iterative_multilabel_assignment(
                new_features,
                split_names,
                remaining_sizes,
                rng,
            )
        else:
            new_assignment = np.empty(0, dtype=np.int16)
        full_assignment = np.full(len(patient_ids), -1, dtype=np.int16)
        for patient, split_name in locked_map.items():
            full_assignment[patient_to_position[patient]] = split_to_index[split_name]
        full_assignment[new_positions] = new_assignment
        if bool(np.any(full_assignment < 0)):
            raise RuntimeError("At least one eligible patient was not assigned")
        score = global_assignment_score(
            full_assignment,
            features,
            len(label_feature_names),
            split_sizes,
            patient_record_counts,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = full_assignment.copy()
            best_trial = trial if new_patients else -1

    assert best_assignment is not None and best_score is not None
    patient_to_split = {
        patient: split_names[int(index)]
        for patient, index in zip(patient_ids, best_assignment)
    }
    split_values = metadata["patient_id"].map(patient_to_split)
    if bool(split_values.isna().any()):
        raise RuntimeError("A control-eligible record has no patient-level split")

    manifest_columns = [
        "row_index",
        "ecg_id",
        "record_uid",
        "patient_id",
        "age",
        "sex",
        "source_template",
        "layout_subtype",
        "patient_id_reliability",
    ]
    manifest = metadata[manifest_columns].copy()
    manifest["split"] = split_values.to_numpy()
    manifest["development_half"] = manifest["split"].isin(["train", "valid"])
    manifest["assignment_source"] = manifest["patient_id"].apply(
        lambda patient: locked_source_label if patient in locked_map else "control_extension"
    )
    for group in CONTROL_GROUPS:
        manifest[f"{group}_labels_native"] = metadata[CONTROL_COLUMNS[group]].apply(
            lambda value: sorted(parse_control_dict(value))
        ).apply(json.dumps)
        for label_index, label in enumerate(cohort.source_vocab[group]):
            manifest[f"target_{group}_{label}"] = cohort.targets[group][
                cohort.eligible_rows, label_index
            ].astype(np.int8)
    manifest = manifest.sort_values("row_index").reset_index(drop=True)

    indices = {
        name: manifest.loc[manifest["split"].eq(name), "row_index"].to_numpy(dtype=np.int64)
        for name in split_names
    }
    patient_sets = {
        name: set(manifest.loc[manifest["split"].eq(name), "patient_id"])
        for name in split_names
    }
    overlap = {
        "train_valid": len(patient_sets["train"] & patient_sets["valid"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "valid_test": len(patient_sets["valid"] & patient_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Patient leakage detected: {overlap}")

    locked_observed = manifest[manifest["assignment_source"].eq(locked_source_label)]
    locked_check = locked_observed["patient_id"].map(locked_map).eq(locked_observed["split"])
    if not bool(locked_check.all()):
        raise RuntimeError("Control split changed at least one locked diagnostic patient assignment")

    distribution = []
    for split_name in split_names:
        part = manifest[manifest["split"].eq(split_name)]
        row: dict[str, Any] = {
            "split": split_name,
            "patients": int(part["patient_id"].nunique()),
            "records": int(len(part)),
            "locked_patients": int(
                part.loc[part["assignment_source"].eq(locked_source_label), "patient_id"].nunique()
            ),
            "extension_patients": int(
                part.loc[part["assignment_source"].eq("control_extension"), "patient_id"].nunique()
            ),
        }
        for group in CONTROL_GROUPS:
            for label in cohort.source_vocab[group]:
                column = f"target_{group}_{label}"
                row[f"{group}_{label}_positive_records"] = int(part[column].sum())
                row[f"{group}_{label}_prevalence"] = float(part[column].mean())
        distribution.append(row)

    summary = {
        "protocol": "50% development and 50% locked test; development is 80% train and 20% validation; overall approximately 40:10:50",
        "unit_of_split": "patient_id",
        "eligible_definition": "at least one mapped diagnostic, form, or rhythm control label",
        "eligible_records": int(len(manifest)),
        "eligible_patients": int(len(patient_ids)),
        "excluded_records_without_any_control_label": int(len(cohort.metadata) - len(manifest)),
        "exact_patient_counts": split_sizes,
        "locked_manifest": str(locked_manifest_path),
        "locked_manifest_kind": (
            "frozen_report_control_split" if frozen_report_control_manifest else "diagnosis_split_with_control_extension"
        ),
        "locked_patient_counts": locked_counts,
        "extension_patient_counts": remaining_sizes,
        "locked_assignment_mismatches": 0,
        "stratification_features": feature_names,
        "stratification_trials": int(trials),
        "selected_trial": int(best_trial),
        "balance_objective": list(best_score),
        "patient_overlap": overlap,
        "splits": distribution,
    }
    return SplitArtifacts(manifest=manifest, indices=indices, summary=summary)


def build_model(
    source_checkpoint_path: Path,
    source_checkpoint: Mapping[str, Any],
    source_vocab: Mapping[str, Sequence[str]],
    tokenizer_checkpoint: Path,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    saved_args = dict(source_checkpoint.get("args", {}))
    tokenizer_checkpoint = resolve_repo_path(tokenizer_checkpoint)
    if not tokenizer_checkpoint.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {tokenizer_checkpoint}")
    model = ReportControlClassifier(
        tokenizer_checkpoint=str(tokenizer_checkpoint),
        vocab=source_vocab,
        seq_length=int(saved_args.get("seq_length", 2250)),
        patch_size=int(saved_args.get("patch_size", 75)),
        latent_ratio=float(saved_args.get("latent_ratio", 0.1)),
        channels=int(saved_args.get("channels", 12)),
        codebook_size=int(saved_args.get("codebook_size", 256)),
        residual_levels=int(saved_args.get("residual_levels", 2)),
        pooling=str(saved_args.get("pooling", "flatten")),
        dropout=float(saved_args.get("dropout", 0.1)),
        quantize=bool(saved_args.get("use_quantized_latents", False)),
        freeze_backbone=False,
    )
    model_state, remapped_keys = remap_legacy_foundation_encoder_keys(
        source_checkpoint["model_state_dict"]
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    provenance = {
        "checkpoint": str(source_checkpoint_path),
        "sha256": sha256_file(source_checkpoint_path),
        "epoch": source_checkpoint.get("epoch"),
        "validation_metrics": source_checkpoint.get("metrics", {}),
        "saved_args": saved_args,
        "tokenizer_checkpoint": str(tokenizer_checkpoint),
        "legacy_foundation_encoder_keys_remapped": remapped_keys,
        "initialization": "strict full-state load; optimizer state deliberately not reused",
    }
    return model, provenance


def active_indices(
    vocab: Mapping[str, Sequence[str]],
    active_labels: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
    return {
        group: np.asarray(
            [index for index, label in enumerate(vocab[group]) if label in set(active_labels[group])],
            dtype=np.int64,
        )
        for group in CONTROL_GROUPS
    }


def inactive_head_snapshot(
    model: nn.Module,
    vocab: Mapping[str, Sequence[str]],
    active_labels: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, torch.Tensor]]:
    snapshot: dict[str, dict[str, torch.Tensor]] = {}
    for group in CONTROL_GROUPS:
        active = set(active_labels[group])
        indices = [index for index, label in enumerate(vocab[group]) if label not in active]
        head = getattr(model, f"{group}_head")
        snapshot[group] = {
            "indices": torch.tensor(indices, dtype=torch.long, device=head.weight.device),
            "weight": head.weight.detach()[indices].clone(),
            "bias": head.bias.detach()[indices].clone(),
        }
    return snapshot


@torch.no_grad()
def restore_inactive_head_rows(
    model: nn.Module,
    snapshot: Mapping[str, Mapping[str, torch.Tensor]],
) -> None:
    for group in CONTROL_GROUPS:
        indices = snapshot[group]["indices"]
        if indices.numel() == 0:
            continue
        head = getattr(model, f"{group}_head")
        head.weight.index_copy_(0, indices, snapshot[group]["weight"])
        head.bias.index_copy_(0, indices, snapshot[group]["bias"])


def build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    backbone = []
    heads = []
    for name, parameter in model.named_parameters():
        (backbone if name.startswith("backbone.") else heads).append(parameter)
    groups = [
        {"params": backbone, "lr": args.backbone_lr},
        {"params": heads, "lr": args.head_lr},
    ]
    summary = [
        {"name": "backbone", "lr": args.backbone_lr, "parameters": sum(p.numel() for p in backbone)},
        {"name": "norm_and_heads", "lr": args.head_lr, "parameters": sum(p.numel() for p in heads)},
    ]
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay), summary


def make_loaders(
    cohort: ControlCohort,
    indices: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    synthetic: SyntheticAugmentation | None = None,
) -> dict[str, DataLoader]:
    datasets = {
        name: AnzhenControlDataset(cohort.signals, cohort.targets, rows)
        for name, rows in indices.items()
    }
    train_dataset: Dataset = datasets["train"]
    if synthetic is not None:
        synthetic_dataset = AnzhenControlDataset(
            synthetic.signals,
            synthetic.targets,
            np.arange(len(synthetic.metadata), dtype=np.int64),
        )
        train_dataset = ConcatDataset([train_dataset, synthetic_dataset])
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_seed,
    }
    return {
        "train": DataLoader(train_dataset, shuffle=True, generator=generator, **common),
        "valid": DataLoader(datasets["valid"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def training_pos_weights(
    targets: Mapping[str, np.ndarray],
    train_rows: np.ndarray,
    active: Mapping[str, np.ndarray],
    cap: float,
    synthetic_targets: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    raw: dict[str, np.ndarray] = {}
    effective: dict[str, np.ndarray] = {}
    for group in CONTROL_GROUPS:
        selected_parts = [targets[group][train_rows][:, active[group]].astype(np.float64)]
        if synthetic_targets is not None:
            selected_parts.append(
                synthetic_targets[group][:, active[group]].astype(np.float64)
            )
        selected = np.concatenate(selected_parts, axis=0)
        positive = selected.sum(axis=0)
        if bool(np.any(positive <= 0)):
            raise ValueError(
                f"Training split has an active {group} label with no positives: {positive.tolist()}"
            )
        negative = len(selected) - positive
        raw[group] = (negative / positive).astype(np.float32)
        effective[group] = (
            np.minimum(raw[group], cap).astype(np.float32) if cap > 0 else raw[group]
        )
    return raw, effective


def group_loss_weights(args: argparse.Namespace) -> dict[str, float]:
    return {
        "diagnostic": float(args.diagnostic_loss_weight),
        "form": float(args.form_loss_weight),
        "rhythm": float(args.rhythm_loss_weight),
    }


def control_loss(
    logits: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    criteria: Mapping[str, nn.Module],
    active: Mapping[str, torch.Tensor],
    group_weights: Mapping[str, float],
) -> torch.Tensor:
    losses = []
    weights = []
    for group in CONTROL_GROUPS:
        weight = float(group_weights[group])
        if weight <= 0 or active[group].numel() == 0:
            continue
        losses.append(criteria[group](logits[group].index_select(1, active[group]), targets[group].index_select(1, active[group])))
        weights.append(weight)
    if not losses:
        raise RuntimeError("No active weighted control losses")
    return sum(loss * weight for loss, weight in zip(losses, weights)) / sum(weights)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criteria: Mapping[str, nn.Module],
    active: Mapping[str, torch.Tensor],
    group_weights: Mapping[str, float],
    inactive_snapshot: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    gradient_clip: float,
    max_batches: int | None,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        signals, diagnostic, form, rhythm, _ = batch
        signals = signals.to(device, non_blocking=True)
        targets = {
            "diagnostic": diagnostic.to(device, non_blocking=True),
            "form": form.to(device, non_blocking=True),
            "rhythm": rhythm.to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(signals)
            loss = control_loss(logits, targets, criteria, active, group_weights)
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        restore_inactive_head_rows(model, inactive_snapshot)
        batch_size = int(signals.size(0))
        loss_sum += float(loss.detach().cpu()) * batch_size
        count += batch_size
    return loss_sum / max(count, 1)


def safe_auroc(target: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else float("nan")


def safe_auprc(target: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(target, probability)) if np.unique(target).size == 2 else float("nan")


def metric_payload(
    targets: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    thresholds: Mapping[str, np.ndarray],
    vocab: Mapping[str, Sequence[str]],
    active_labels_map: Mapping[str, Sequence[str]],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    metrics: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    prompt_auc: list[float] = []
    prompt_auprc: list[float] = []
    prompt_f1: list[float] = []
    all_auc: list[float] = []
    all_auprc: list[float] = []
    all_f1: list[float] = []
    for group in CONTROL_GROUPS:
        active_set = set(active_labels_map[group])
        group_auc = []
        group_auprc = []
        group_f1 = []
        for label_index, label in enumerate(vocab[group]):
            if label not in active_set:
                continue
            target = targets[group][:, label_index].astype(np.int8)
            probability = probabilities[group][:, label_index]
            prediction = (probability >= thresholds[group][label_index]).astype(np.int8)
            auc = safe_auroc(target, probability)
            auprc = safe_auprc(target, probability)
            f1 = float(f1_score(target, prediction, zero_division=0))
            row = {
                "group": group,
                "label": label,
                "support": int(target.sum()),
                "prevalence": float(target.mean()),
                "threshold": float(thresholds[group][label_index]),
                "auroc": auc,
                "auprc": auprc,
                "f1": f1,
            }
            rows.append(row)
            if math.isfinite(auc):
                group_auc.append(auc)
                all_auc.append(auc)
                if group in {"form", "rhythm"}:
                    prompt_auc.append(auc)
            if math.isfinite(auprc):
                group_auprc.append(auprc)
                all_auprc.append(auprc)
                if group in {"form", "rhythm"}:
                    prompt_auprc.append(auprc)
            group_f1.append(f1)
            all_f1.append(f1)
            if group in {"form", "rhythm"}:
                prompt_f1.append(f1)
        metrics[f"{group}_macro_auroc"] = float(np.mean(group_auc)) if group_auc else float("nan")
        metrics[f"{group}_macro_auprc"] = float(np.mean(group_auprc)) if group_auprc else float("nan")
        metrics[f"{group}_macro_f1"] = float(np.mean(group_f1)) if group_f1 else float("nan")
    metrics["all_active_macro_auroc"] = float(np.mean(all_auc)) if all_auc else float("nan")
    metrics["all_active_macro_auprc"] = float(np.mean(all_auprc)) if all_auprc else float("nan")
    metrics["all_active_macro_f1"] = float(np.mean(all_f1)) if all_f1 else float("nan")
    metrics["prompt_macro_auroc"] = float(np.mean(prompt_auc)) if prompt_auc else float("nan")
    metrics["prompt_macro_auprc"] = float(np.mean(prompt_auprc)) if prompt_auprc else float("nan")
    metrics["prompt_macro_f1"] = float(np.mean(prompt_f1)) if prompt_f1 else float("nan")
    return metrics, rows


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    vocab: Mapping[str, Sequence[str]],
    active_labels_map: Mapping[str, Sequence[str]],
    active: Mapping[str, torch.Tensor],
    group_weights: Mapping[str, float],
    max_batches: int | None,
) -> dict[str, Any]:
    model.eval()
    criteria = {group: nn.BCEWithLogitsLoss() for group in CONTROL_GROUPS}
    logits_buffer = {group: [] for group in CONTROL_GROUPS}
    targets_buffer = {group: [] for group in CONTROL_GROUPS}
    rows_buffer = []
    loss_sum = 0.0
    count = 0
    for batch_index, batch in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        signals, diagnostic, form, rhythm, rows = batch
        signals = signals.to(device, non_blocking=True)
        targets_device = {
            "diagnostic": diagnostic.to(device, non_blocking=True),
            "form": form.to(device, non_blocking=True),
            "rhythm": rhythm.to(device, non_blocking=True),
        }
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(signals)
            loss = control_loss(logits, targets_device, criteria, active, group_weights)
        batch_size = int(signals.size(0))
        loss_sum += float(loss.detach().cpu()) * batch_size
        count += batch_size
        for group in CONTROL_GROUPS:
            logits_buffer[group].append(logits[group].float().cpu())
            targets_buffer[group].append(targets_device[group].float().cpu())
        rows_buffer.append(rows.long().cpu())
    if not rows_buffer:
        raise RuntimeError("Evaluation produced no batches")
    logits_np = {group: torch.cat(logits_buffer[group]).numpy() for group in CONTROL_GROUPS}
    targets_np = {
        group: torch.cat(targets_buffer[group]).numpy().astype(np.int8)
        for group in CONTROL_GROUPS
    }
    probabilities = {group: sigmoid(logits_np[group]) for group in CONTROL_GROUPS}
    fixed_thresholds = {
        group: np.full(len(vocab[group]), 0.5, dtype=np.float32) for group in CONTROL_GROUPS
    }
    metrics, per_class = metric_payload(
        targets_np,
        probabilities,
        fixed_thresholds,
        vocab,
        active_labels_map,
    )
    return {
        "loss": loss_sum / max(count, 1),
        "logits": logits_np,
        "targets": targets_np,
        "probabilities": probabilities,
        "rows": torch.cat(rows_buffer).numpy().astype(np.int64),
        "metrics_fixed_0p5": metrics,
        "per_class_fixed_0p5": per_class,
    }


def validation_thresholds(
    targets: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    vocab: Mapping[str, Sequence[str]],
    active_labels_map: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
    grid = np.linspace(0.01, 0.99, 99, dtype=np.float32)
    outputs = {
        group: np.full(len(vocab[group]), 0.5, dtype=np.float32) for group in CONTROL_GROUPS
    }
    for group in CONTROL_GROUPS:
        active_set = set(active_labels_map[group])
        for label_index, label in enumerate(vocab[group]):
            if label not in active_set:
                continue
            target = targets[group][:, label_index]
            scores = []
            for threshold in grid:
                prediction = probabilities[group][:, label_index] >= threshold
                scores.append(float(f1_score(target, prediction, zero_division=0)))
            scores_array = np.asarray(scores)
            best = np.flatnonzero(np.isclose(scores_array, scores_array.max()))
            closest = best[np.argmin(np.abs(grid[best] - 0.5))]
            outputs[group][label_index] = grid[closest]
    return outputs


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: Mapping[str, Any],
    vocab: Mapping[str, Sequence[str]],
    active_labels_map: Mapping[str, Sequence[str]],
    args: argparse.Namespace,
    source_checkpoint: Path,
    tokenizer_checkpoint: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "control_vocab": {group: list(vocab[group]) for group in CONTROL_GROUPS},
            "active_target_labels": {
                group: list(active_labels_map[group]) for group in CONTROL_GROUPS
            },
            "metrics": dict(metrics),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "source_checkpoint": str(source_checkpoint),
            "tokenizer_checkpoint": tokenizer_checkpoint,
        },
        path,
    )


def prediction_frame(
    manifest: pd.DataFrame,
    result: Mapping[str, Any],
    vocab: Mapping[str, Sequence[str]],
    thresholds: Mapping[str, np.ndarray],
    model_stage: str,
) -> pd.DataFrame:
    by_row = manifest.set_index("row_index")
    frame = by_row.loc[result["rows"]].reset_index()[
        ["row_index", "ecg_id", "record_uid", "patient_id", "split", "layout_subtype"]
    ].copy()
    frame.insert(0, "model_stage", model_stage)
    for group in CONTROL_GROUPS:
        for label_index, label in enumerate(vocab[group]):
            stem = f"{group}_{label}"
            frame[f"target_{stem}"] = result["targets"][group][:, label_index]
            frame[f"prob_{stem}"] = result["probabilities"][group][:, label_index]
            frame[f"pred_fixed_0p5_{stem}"] = (
                result["probabilities"][group][:, label_index] >= 0.5
            ).astype(np.int8)
            frame[f"pred_val_threshold_{stem}"] = (
                result["probabilities"][group][:, label_index]
                >= thresholds[group][label_index]
            ).astype(np.int8)
    return frame


def clustered_bootstrap(
    result: Mapping[str, Any],
    patient_ids: np.ndarray,
    vocab: Mapping[str, Sequence[str]],
    active_labels_map: Mapping[str, Sequence[str]],
    thresholds: Mapping[str, np.ndarray],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    unique_patients = np.unique(patient_ids)
    patient_rows = {patient: np.flatnonzero(patient_ids == patient) for patient in unique_patients}
    metric_names = [
        "prompt_macro_auroc",
        "prompt_macro_auprc",
        "prompt_macro_f1",
        "diagnostic_macro_auroc",
        "form_macro_auroc",
        "rhythm_macro_auroc",
    ]
    samples = {name: [] for name in metric_names}
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        selected = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        rows = np.concatenate([patient_rows[patient] for patient in selected])
        metrics, _ = metric_payload(
            {group: result["targets"][group][rows] for group in CONTROL_GROUPS},
            {group: result["probabilities"][group][rows] for group in CONTROL_GROUPS},
            thresholds,
            vocab,
            active_labels_map,
        )
        for name in metric_names:
            value = float(metrics[name])
            if math.isfinite(value):
                samples[name].append(value)
    output: dict[str, Any] = {
        "unit": "patient_id cluster",
        "requested_replicates": int(replicates),
        "patients": int(len(unique_patients)),
    }
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        output[name] = {
            "valid_replicates": int(array.size),
            "median": float(np.median(array)) if array.size else float("nan"),
            "ci_95_low": float(np.quantile(array, 0.025)) if array.size else float("nan"),
            "ci_95_high": float(np.quantile(array, 0.975)) if array.size else float("nan"),
        }
    return output


def write_results_zh(path: Path, summary: Mapping[str, Any]) -> None:
    source = summary["source_zero_shot_test_fixed_0p5"]
    final = summary["finetuned_test_fixed_0p5_primary"]
    augmentation = summary.get("training_augmentation")
    model_name = "安贞 50:50 真实数据适配"
    training_line = "- 训练集仅包含真实训练患者记录。"
    if augmentation:
        model_name = "安贞 50:50 真实＋派生数据适配"
        training_line = (
            f"- Fine-tune 使用 {augmentation['real_train_records']} 条真实训练记录与 "
            f"{augmentation['synthetic_train_records']} 条训练派生记录；验证集和测试集保持纯真实。"
        )
    lines = [
        "# 安贞 Report Control Head 适配结果",
        "",
        "## 实验协议",
        "",
        "- 患者级 50:50：50% development（其中 80%/20% 为训练/验证），50% 锁定测试。",
        "- 复用既有诊断实验的患者归属；新增 control-only 患者仅补足各 split，患者交集为 0。",
        "- 保持 PTB-XL TRACE control vocabulary 不变，以便 checkpoint 可直接接入现有报告生成器。",
        "- checkpoint 仅由验证集 prompt macro-AUROC 选择；测试集在训练和选模结束后首次访问。",
        "- 主结果使用固定阈值 0.5；验证集逐类阈值仅作为次要分析。",
        training_line,
        "",
        "## 锁定测试集主结果（固定阈值 0.5）",
        "",
        "| 模型 | Prompt macro-AUROC | Prompt macro-AUPRC | Prompt macro-F1 |",
        "|---|---:|---:|---:|",
        f"| PTB-XL source（zero-shot） | {source['prompt_macro_auroc']:.4f} | {source['prompt_macro_auprc']:.4f} | {source['prompt_macro_f1']:.4f} |",
        f"| {model_name} | {final['prompt_macro_auroc']:.4f} | {final['prompt_macro_auprc']:.4f} | {final['prompt_macro_f1']:.4f} |",
        "",
        "注：Prompt 指实际进入 TRACE report prompt 的 form/rhythm 分支；diagnostic 分支在正式推理时由 Stage II 诊断模型提供。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    started = time.time()
    args.data_root = resolve_repo_path(args.data_root)
    source_checkpoint_path = resolve_repo_path(args.source_checkpoint)
    locked_manifest_path = resolve_repo_path(
        args.locked_split_manifest
        if args.locked_split_manifest is not None
        else args.data_root / "trace_compat/paper_report_control_split_seed42.csv"
    )
    output_dir = resolve_repo_path(args.output_dir)
    synthetic_root = (
        resolve_repo_path(args.synthetic_root) if args.synthetic_root is not None else None
    )
    if not source_checkpoint_path.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint_path}")
    if not locked_manifest_path.exists():
        raise FileNotFoundError(f"Locked split manifest not found: {locked_manifest_path}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}; choose another path or pass --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu")
    source_vocab = source_vocab_from_checkpoint(source_checkpoint)
    cohort = load_control_cohort(args, source_vocab)
    split = build_patient_split(
        cohort,
        locked_manifest_path,
        args.seed,
        args.split_trials,
        args.split_label_min_count,
    )
    split.manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    write_json(output_dir / "split_summary.json", split.summary)
    pd.DataFrame(split.summary["splits"]).to_csv(
        output_dir / "split_distribution.csv", index=False
    )
    synthetic = (
        load_synthetic_augmentation(
            synthetic_root,
            source_vocab,
            split,
            args.seq_length,
            args.latent_ratio,
        )
        if synthetic_root is not None
        else None
    )
    if synthetic is not None:
        write_json(output_dir / "synthetic_augmentation_audit.json", synthetic.provenance)

    transfer_audit = {
        "compatibility_policy": "preserve the exact PTB-XL TRACE vocabulary and row order so the checkpoint remains compatible with the existing report-generator control tokens",
        "source_checkpoint": str(source_checkpoint_path),
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "source_vocab": source_vocab,
        "target_checkpoint_vocab": source_vocab,
        "strict_full_state_initialization": True,
        "active_labels_with_anzhen_support": cohort.active_labels,
        "inactive_rows_restored_after_every_optimizer_step": cohort.inactive_labels,
        "anzhen_native_label_counts": cohort.native_label_counts,
        "target_only_labels_not_added_to_generator_compatible_head": {
            group: sorted(set(cohort.native_label_counts[group]) - set(source_vocab[group]))
            for group in CONTROL_GROUPS
        },
    }
    write_json(output_dir / "label_transfer_audit.json", transfer_audit)
    write_json(
        output_dir / "control_vocab.json",
        {group: list(source_vocab[group]) for group in CONTROL_GROUPS},
    )
    print(json.dumps(json_safe(split.summary), ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(json_safe(transfer_audit), ensure_ascii=False, indent=2), flush=True)
    if args.split_only:
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_amp = device.type == "cuda" and not args.disable_amp
    loaders = make_loaders(cohort, split.indices, args, device, synthetic)
    model, source_provenance = build_model(
        source_checkpoint_path,
        source_checkpoint,
        source_vocab,
        args.tokenizer_checkpoint,
        device,
    )
    optimizer, optimizer_summary = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    active_np = active_indices(source_vocab, cohort.active_labels)
    active_device = {
        group: torch.tensor(active_np[group], dtype=torch.long, device=device)
        for group in CONTROL_GROUPS
    }
    raw_pos_weight, effective_pos_weight = training_pos_weights(
        cohort.targets,
        split.indices["train"],
        active_np,
        args.pos_weight_cap,
        synthetic.targets if synthetic is not None else None,
    )
    criteria = {
        group: nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(effective_pos_weight[group], dtype=torch.float32, device=device)
        )
        for group in CONTROL_GROUPS
    }
    weights = group_loss_weights(args)
    inactive_snapshot = inactive_head_snapshot(model, source_vocab, cohort.active_labels)
    scaler = GradScaler(enabled=use_amp)
    tokenizer_checkpoint = source_provenance["tokenizer_checkpoint"]

    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Anzhen adaptation of the TRACE report-control head",
        "split": split.summary,
        "source": source_provenance,
        "label_transfer": transfer_audit,
        "training_data": {
            "real_train_records": int(len(split.indices["train"])),
            "real_validation_records": int(len(split.indices["valid"])),
            "real_test_records": int(len(split.indices["test"])),
            "synthetic_train_records": int(len(synthetic.metadata)) if synthetic is not None else 0,
            "total_training_instances_per_epoch": int(len(loaders["train"].dataset)),
            "synthetic_training_fraction": (
                float(len(synthetic.metadata) / len(loaders["train"].dataset))
                if synthetic is not None
                else 0.0
            ),
            "validation_and_test_are_real_only": True,
            "synthetic_new_independent_patients": 0,
            "synthetic_provenance": synthetic.provenance if synthetic is not None else None,
        },
        "fine_tuning": {
            "trainable_scope": "all backbone parameters, LayerNorm, and active output rows",
            "inactive_output_policy": "source-only or support<minimum rows are restored exactly after every optimizer step",
            "optimizer": "AdamW with fresh optimizer state and discriminative learning rates",
            "optimizer_groups": optimizer_summary,
            "weight_decay": args.weight_decay,
            "loss": "group-balanced, class-weighted BCE on active labels only",
            "group_loss_weights": weights,
            "raw_pos_weight": {
                group: dict(zip(cohort.active_labels[group], raw_pos_weight[group].tolist()))
                for group in CONTROL_GROUPS
            },
            "effective_pos_weight": {
                group: dict(zip(cohort.active_labels[group], effective_pos_weight[group].tolist()))
                for group in CONTROL_GROUPS
            },
            "pos_weight_cap": args.pos_weight_cap,
            "checkpoint_selection": f"validation {args.target_metric}; form/rhythm active labels only",
            "test_use_during_training_or_selection": "none",
            "primary_threshold": 0.5,
            "secondary_thresholds": "per-class F1 thresholds selected on validation only",
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "minimum_epochs": args.min_epochs,
            "seed": args.seed,
            "amp": use_amp,
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    write_json(output_dir / "protocol.json", protocol)

    history_path = output_dir / "history.jsonl"
    best_value = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(args.epochs):
        train_loss = train_epoch(
            model,
            loaders["train"],
            optimizer,
            criteria,
            active_device,
            weights,
            inactive_snapshot,
            device,
            scaler,
            use_amp,
            args.gradient_clip,
            args.max_train_batches,
        )
        valid_result = evaluate_loader(
            model,
            loaders["valid"],
            device,
            use_amp,
            source_vocab,
            cohort.active_labels,
            active_device,
            weights,
            args.max_eval_batches,
        )
        metric_value = float(valid_result["metrics_fixed_0p5"][args.target_metric])
        row = {
            "epoch": epoch,
            "train_loss_weighted": train_loss,
            "valid_loss_unweighted": valid_result["loss"],
            "selection_metric": args.target_metric,
            "selection_value": metric_value,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            **valid_result["metrics_fixed_0p5"],
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        print(json.dumps(json_safe(row), ensure_ascii=False), flush=True)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            row,
            source_vocab,
            cohort.active_labels,
            args,
            source_checkpoint_path,
            tokenizer_checkpoint,
        )
        improved = math.isfinite(metric_value) and metric_value > best_value + 1e-8
        if best_epoch < 0 or improved:
            best_value = metric_value
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                row,
                source_vocab,
                cohort.active_labels,
                args,
                source_checkpoint_path,
                tokenizer_checkpoint,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if epoch + 1 >= args.min_epochs and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}", flush=True)
            break

    if args.smoke_test:
        write_json(
            output_dir / "smoke_test_complete.json",
            {
                "status": "passed",
                "best_epoch": int(best_epoch),
                "best_validation_selection_value": float(best_value),
                "locked_test_accessed": False,
                "max_train_batches": args.max_train_batches,
                "max_eval_batches": args.max_eval_batches,
            },
        )
        print("Smoke test passed; locked test set was not accessed.", flush=True)
        return

    # The locked test loader is first iterated only after validation-only model selection.
    source_model_state, _ = remap_legacy_foundation_encoder_keys(
        source_checkpoint["model_state_dict"]
    )
    model.load_state_dict(source_model_state, strict=True)
    source_test = evaluate_loader(
        model,
        loaders["test"],
        device,
        use_amp,
        source_vocab,
        cohort.active_labels,
        active_device,
        weights,
        None,
    )
    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    best_model_state, _ = remap_legacy_foundation_encoder_keys(
        best_checkpoint["model_state_dict"]
    )
    model.load_state_dict(best_model_state, strict=True)
    final_valid = evaluate_loader(
        model,
        loaders["valid"],
        device,
        use_amp,
        source_vocab,
        cohort.active_labels,
        active_device,
        weights,
        None,
    )
    selected_thresholds = validation_thresholds(
        final_valid["targets"],
        final_valid["probabilities"],
        source_vocab,
        cohort.active_labels,
    )
    final_test = evaluate_loader(
        model,
        loaders["test"],
        device,
        use_amp,
        source_vocab,
        cohort.active_labels,
        active_device,
        weights,
        None,
    )
    fixed_thresholds = {
        group: np.full(len(source_vocab[group]), 0.5, dtype=np.float32)
        for group in CONTROL_GROUPS
    }
    source_metrics, source_per_class = metric_payload(
        source_test["targets"],
        source_test["probabilities"],
        fixed_thresholds,
        source_vocab,
        cohort.active_labels,
    )
    final_fixed_metrics, final_fixed_per_class = metric_payload(
        final_test["targets"],
        final_test["probabilities"],
        fixed_thresholds,
        source_vocab,
        cohort.active_labels,
    )
    final_selected_metrics, final_selected_per_class = metric_payload(
        final_test["targets"],
        final_test["probabilities"],
        selected_thresholds,
        source_vocab,
        cohort.active_labels,
    )
    per_class_frames = []
    for stage, scheme, rows in [
        ("source_ptbxl_zero_shot", "fixed_0p5", source_per_class),
        ("anzhen_finetuned", "fixed_0p5", final_fixed_per_class),
        ("anzhen_finetuned", "validation_selected_per_class", final_selected_per_class),
    ]:
        frame = pd.DataFrame(rows)
        frame.insert(0, "model_stage", stage)
        frame.insert(1, "threshold_scheme", scheme)
        per_class_frames.append(frame)
    pd.concat(per_class_frames, ignore_index=True).to_csv(
        output_dir / "test_per_class_metrics.csv", index=False
    )

    predictions = pd.concat(
        [
            prediction_frame(
                split.manifest,
                source_test,
                source_vocab,
                selected_thresholds,
                "source_ptbxl_zero_shot",
            ),
            prediction_frame(
                split.manifest,
                final_test,
                source_vocab,
                selected_thresholds,
                "anzhen_finetuned",
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        row_index=final_test["rows"],
        **{
            f"targets_{group}": final_test["targets"][group]
            for group in CONTROL_GROUPS
        },
        **{
            f"source_probabilities_{group}": source_test["probabilities"][group].astype(np.float32)
            for group in CONTROL_GROUPS
        },
        **{
            f"finetuned_probabilities_{group}": final_test["probabilities"][group].astype(np.float32)
            for group in CONTROL_GROUPS
        },
        **{
            f"validation_selected_thresholds_{group}": selected_thresholds[group]
            for group in CONTROL_GROUPS
        },
    )

    test_patient_ids = (
        split.manifest.set_index("row_index")
        .loc[final_test["rows"], "patient_id"]
        .astype(str)
        .to_numpy()
    )
    bootstrap_fixed = clustered_bootstrap(
        final_test,
        test_patient_ids,
        source_vocab,
        cohort.active_labels,
        fixed_thresholds,
        args.bootstrap_replicates,
        args.seed,
    )
    bootstrap_selected = clustered_bootstrap(
        final_test,
        test_patient_ids,
        source_vocab,
        cohort.active_labels,
        selected_thresholds,
        args.bootstrap_replicates,
        args.seed + 1,
    )
    delta = {
        key: float(final_fixed_metrics[key] - source_metrics[key])
        for key in ("prompt_macro_auroc", "prompt_macro_auprc", "prompt_macro_f1")
    }
    threshold_payload = {
        group: dict(zip(source_vocab[group], selected_thresholds[group].tolist()))
        for group in CONTROL_GROUPS
    }
    summary = {
        "protocol": split.summary["protocol"],
        "source_checkpoint": source_provenance,
        "generator_compatibility": "exact PTB-XL TRACE control vocabulary and row order preserved",
        "active_labels": cohort.active_labels,
        "inactive_preserved_labels": cohort.inactive_labels,
        "training_augmentation": (
            {
                "real_train_records": int(len(split.indices["train"])),
                "synthetic_train_records": int(len(synthetic.metadata)),
                "synthetic_parent_patients": synthetic.provenance["parent_patients"],
                "synthetic_training_fraction": float(
                    len(synthetic.metadata) / len(loaders["train"].dataset)
                ),
                "validation_and_test_are_real_only": True,
                "new_independent_patients": 0,
                "provenance": synthetic.provenance,
            }
            if synthetic is not None
            else None
        ),
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_metrics": best_checkpoint.get("metrics", {}),
        "test_access": "only after completion of training and validation-based checkpoint selection",
        "test_patients": int(len(np.unique(test_patient_ids))),
        "test_records": int(len(final_test["rows"])),
        "validation_selected_thresholds": threshold_payload,
        "source_zero_shot_test_fixed_0p5": source_metrics,
        "finetuned_test_fixed_0p5_primary": final_fixed_metrics,
        "finetuned_test_validation_selected_thresholds_secondary": final_selected_metrics,
        "finetuned_minus_source_fixed_0p5": delta,
        "finetuned_fixed_0p5_patient_bootstrap": bootstrap_fixed,
        "finetuned_validation_threshold_patient_bootstrap": bootstrap_selected,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "test_metrics.json", summary)
    write_results_zh(output_dir / "RESULTS_zh.md", summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
