#!/usr/bin/env python3
"""Locked external evaluation of PTB-XL-trained TRACE on the Anzhen cohort.

This entry point is intentionally inference-only.  It never constructs an
optimizer, never updates model parameters, and never uses Anzhen labels for
checkpoint or threshold selection.  PTB-XL validation-derived thresholds are
loaded from an existing audit artifact and applied unchanged.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.checkpoint_compat import remap_legacy_foundation_encoder_keys
from utils.downstream_core import TRACEDiagnosisClassifier as BaseClassifier
from utils.downstream_trace import (
    PTBXL_SUPERCLASSES,
    TRACEDiagnosisClassifier as TRACEClassifier,
    slice_signal_window,
)


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data/anzhen_external"
DEFAULT_TRACE_CHECKPOINT = REPO_ROOT / "ckpt/ptbxl/downstream_refined/best.pt"
DEFAULT_BASE_CHECKPOINT = REPO_ROOT / "ckpt/ptbxl/downstream_base/best.pt"
DEFAULT_TOKENIZER_CHECKPOINT = (
    REPO_ROOT
    / "ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth"
)
DEFAULT_THRESHOLD_CONFIG = REPO_ROOT / "data/ptbxl_validation_thresholds.json"
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


@dataclass(frozen=True)
class Cohort:
    signals: np.ndarray
    valid_mask: np.ndarray
    metadata: pd.DataFrame
    subsets: pd.DataFrame
    targets: np.ndarray
    selected_rows: np.ndarray
    class_names: list[str]


class ExternalInferenceDataset(Dataset):
    """Signal-only dataset; labels are deliberately absent from inference."""

    def __init__(self, signals: np.ndarray, selected_rows: np.ndarray):
        self.signals = signals
        self.selected_rows = np.asarray(selected_rows, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.selected_rows.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = int(self.selected_rows[index])
        # np.load(..., mmap_mode="r") returns a read-only view; copy before
        # torch.from_numpy so PyTorch never receives a non-writable buffer.
        signal = np.array(self.signals[row], dtype=np.float32, copy=True)
        return torch.from_numpy(signal), row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked zero-shot external diagnosis evaluation on the Anzhen ECG cohort"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results/anzhen_external_ptbxl_v14",
    )
    parser.add_argument("--trace-checkpoint", type=Path, default=DEFAULT_TRACE_CHECKPOINT)
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=Path,
        default=DEFAULT_TOKENIZER_CHECKPOINT,
        help="Explicit tokenizer checkpoint used by both locked diagnosis models.",
    )
    parser.add_argument("--threshold-config", type=Path, default=DEFAULT_THRESHOLD_CONFIG)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("base", "trace"),
        default=("base", "trace"),
        help="Locked PTB-XL models to evaluate on the same external cohort.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--seq-length", type=int, default=2250)
    parser.add_argument("--latent-ratio", type=float, default=0.1)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None, help="Smoke-test only; selects the first N rows.")
    parser.add_argument("--skip-integrity-hashes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_bool_column(series: pd.Series, name: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    unknown = ~normalized.isin(["true", "false"])
    if bool(unknown.any()):
        examples = series[unknown].head().tolist()
        raise ValueError(f"Column {name} contains non-Boolean values: {examples}")
    return normalized.eq("true").to_numpy(dtype=bool)


def load_cohort(args: argparse.Namespace) -> Cohort:
    trace_root = args.data_root / "trace_compat"
    paths = {
        "signals": trace_root / "full_data.npy",
        "valid_mask": trace_root / "valid_mask.npy",
        "metadata": trace_root / "ptbxl_compat_database.csv",
        "subsets": trace_root / "evaluation_subsets.csv",
        "row_index": trace_root / "row_index.csv",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Anzhen release artifacts: {missing}")

    metadata = pd.read_csv(paths["metadata"])
    subsets = pd.read_csv(paths["subsets"])
    row_index = pd.read_csv(paths["row_index"])
    raw_signals = np.load(paths["signals"], mmap_mode="r")
    raw_valid_mask = np.load(paths["valid_mask"], mmap_mode="r")

    n_records = len(metadata)
    if not (len(subsets) == len(row_index) == raw_signals.shape[0] == raw_valid_mask.shape[0] == n_records):
        raise ValueError(
            "Row-count mismatch among signals, validity mask, metadata, subsets, and row index: "
            f"signals={raw_signals.shape[0]}, mask={raw_valid_mask.shape[0]}, metadata={len(metadata)}, "
            f"subsets={len(subsets)}, row_index={len(row_index)}"
        )
    if tuple(raw_signals.shape[1:]) != (12, 2500):
        raise ValueError(f"Expected signals shaped (N, 12, 2500), got {raw_signals.shape}")
    if raw_valid_mask.shape != raw_signals.shape:
        raise ValueError(f"Validity-mask shape {raw_valid_mask.shape} != signal shape {raw_signals.shape}")
    expected_rows = np.arange(n_records, dtype=np.int64)
    if not np.array_equal(subsets["row_index"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError("evaluation_subsets.csv row_index is not contiguous and row-aligned")
    if not np.array_equal(row_index["row_index"].to_numpy(dtype=np.int64), expected_rows):
        raise ValueError("row_index.csv row_index is not contiguous and row-aligned")
    for table_name, table in (("subsets", subsets), ("row_index", row_index)):
        if not np.array_equal(table["ecg_id"].to_numpy(), metadata["ecg_id"].to_numpy()):
            raise ValueError(f"ecg_id mismatch between metadata and {table_name}")
        if not np.array_equal(table["record_uid"].astype(str).to_numpy(), metadata["record_uid"].astype(str).to_numpy()):
            raise ValueError(f"record_uid mismatch between metadata and {table_name}")

    if set(metadata["recommended_split"].astype(str)) != {"external_test"}:
        raise ValueError("All Anzhen records must remain marked recommended_split=external_test")
    if set(pd.to_numeric(metadata["strat_fold"], errors="raise").astype(int)) != {10}:
        raise ValueError("All Anzhen records must retain compatibility-only strat_fold=10")
    if metadata["patient_id"].isna().any():
        raise ValueError("Missing patient identifiers prevent patient-clustered evaluation")

    signals = slice_signal_window(raw_signals, seq_length=args.seq_length, latent_ratio=args.latent_ratio)
    valid_mask = slice_signal_window(raw_valid_mask, seq_length=args.seq_length, latent_ratio=args.latent_ratio)
    if not np.isfinite(signals).all():
        raise ValueError("TRACE compatibility signals contain NaN or infinite values")

    class_names = list(PTBXL_SUPERCLASSES)
    class_to_index = {name: index for index, name in enumerate(class_names)}
    eligible = parse_bool_column(subsets["has_five_superclass_target"], "has_five_superclass_target")
    targets = np.zeros((n_records, len(class_names)), dtype=np.int8)
    for row, value in enumerate(subsets["diagnostic_superclasses"]):
        labels = ast.literal_eval(value) if isinstance(value, str) and value.strip() else []
        unknown = sorted(set(labels) - set(class_names))
        if unknown:
            raise ValueError(f"Unknown superclass labels at row {row}: {unknown}")
        for label in labels:
            targets[row, class_to_index[label]] = 1
    if not np.array_equal(targets.sum(axis=1) > 0, eligible):
        raise ValueError("Five-superclass eligibility flag disagrees with parsed reference targets")

    selected_rows = expected_rows
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        selected_rows = expected_rows[: min(args.max_samples, n_records)]

    return Cohort(
        signals=signals,
        valid_mask=valid_mask,
        metadata=metadata,
        subsets=subsets,
        targets=targets,
        selected_rows=selected_rows,
        class_names=class_names,
    )


def load_threshold_schemes(path: Path, class_names: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    datasets = payload.get("datasets", {})
    ptb = datasets.get("PTB-XL") or datasets.get("ptbxl")
    if not ptb:
        raise ValueError(f"PTB-XL threshold entry not found in {path}")
    if list(ptb.get("class_names", [])) != list(class_names):
        raise ValueError(
            f"Threshold class order {ptb.get('class_names')} does not match checkpoint order {list(class_names)}"
        )
    fixed = float(payload.get("protocol", {}).get("main_threshold", 0.5))
    global_threshold = float(ptb["global_val"]["threshold"])
    per_class = np.asarray(ptb["per_class_val"]["thresholds"], dtype=np.float32)
    if per_class.shape != (len(class_names),):
        raise ValueError(f"Expected {len(class_names)} per-class thresholds, got {per_class.shape}")
    schemes = {
        "fixed_0p5_primary": np.full(len(class_names), fixed, dtype=np.float32),
        "global_ptbxl_validation": np.full(len(class_names), global_threshold, dtype=np.float32),
        "per_class_ptbxl_validation": per_class,
    }
    provenance = {
        "source": str(path),
        "selection_dataset": "PTB-XL validation split only",
        "anzhen_label_use": "none for threshold selection",
        "schemes": {name: values.tolist() for name, values in schemes.items()},
    }
    return schemes, provenance


def checkpoint_model(
    *,
    kind: str,
    checkpoint_path: Path,
    tokenizer_checkpoint: Path,
    expected_class_names: Sequence[str],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint_path = resolve_repo_path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = list(checkpoint.get("class_names", []))
    if class_names != list(expected_class_names):
        raise ValueError(
            f"{kind} checkpoint class order {class_names} != expected {list(expected_class_names)}"
        )
    saved_args = dict(checkpoint.get("args", {}))
    tokenizer_checkpoint = resolve_repo_path(tokenizer_checkpoint)
    if not tokenizer_checkpoint.is_file():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {tokenizer_checkpoint}")
    common = {
        "tokenizer_checkpoint": str(tokenizer_checkpoint),
        "num_classes": len(class_names),
        "seq_length": int(saved_args.get("seq_length", 2250)),
        "patch_size": int(saved_args.get("patch_size", 75)),
        "latent_ratio": float(saved_args.get("latent_ratio", 0.1)),
        "channels": int(saved_args.get("channels", 12)),
        "codebook_size": int(saved_args.get("codebook_size", 256)),
        "residual_levels": int(saved_args.get("residual_levels", 2)),
        "pooling": str(saved_args.get("pooling", "flatten")),
        "dropout": float(saved_args.get("dropout", 0.1)),
        "quantize": bool(saved_args.get("use_quantized_latents", False)),
        "freeze_backbone": True,
    }
    if kind == "base":
        model = BaseClassifier(**common)
    elif kind == "trace":
        model = TRACEClassifier(
            **common,
            base_downstream_checkpoint="",
            freeze_base_branch=False,
        )
    else:
        raise ValueError(f"Unsupported model kind: {kind}")
    model_state_dict, remapped_keys = remap_legacy_foundation_encoder_keys(
        checkpoint["model_state_dict"]
    )
    model.load_state_dict(model_state_dict, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    info = {
        "kind": kind,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_metrics": checkpoint.get("metrics", {}),
        "checkpoint_saved_args": saved_args,
        "legacy_foundation_encoder_keys_remapped": remapped_keys,
        "checkpoint_saved_tokenizer_checkpoint": saved_args.get("tokenizer_checkpoint"),
        "tokenizer_checkpoint": str(tokenizer_checkpoint),
        "class_names": class_names,
        "trainable_parameters_during_external_evaluation": 0,
    }
    return model, info


@torch.inference_mode()
def collect_logits(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    use_amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    logits_buffer: list[torch.Tensor] = []
    rows_buffer: list[torch.Tensor] = []
    for signals, rows in tqdm(loader, desc="external inference", leave=False):
        signals = signals.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(signals)
        logits_buffer.append(logits.cpu())
        rows_buffer.append(rows.cpu())
    return torch.cat(logits_buffer).numpy(), torch.cat(rows_buffer).numpy().astype(np.int64)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def safe_auroc(target: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(target, probability)) if np.unique(target).size == 2 else float("nan")


def safe_auprc(target: np.ndarray, probability: np.ndarray) -> float:
    return float(average_precision_score(target, probability)) if np.unique(target).size == 2 else float("nan")


def safe_nanmean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def class_metric_rows(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str],
) -> list[dict[str, Any]]:
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)
    rows = []
    for index, class_name in enumerate(class_names):
        target = targets[:, index].astype(np.int8)
        pred = predictions[:, index]
        tp = int(np.sum((target == 1) & (pred == 1)))
        fp = int(np.sum((target == 0) & (pred == 1)))
        tn = int(np.sum((target == 0) & (pred == 0)))
        fn = int(np.sum((target == 1) & (pred == 0)))
        rows.append(
            {
                "class": class_name,
                "threshold": float(thresholds[index]),
                "support_positive": int(target.sum()),
                "support_negative": int((1 - target).sum()),
                "prevalence": float(target.mean()),
                "auroc": safe_auroc(target, probabilities[:, index]),
                "auprc": safe_auprc(target, probabilities[:, index]),
                "precision": float(precision_score(target, pred, zero_division=0)),
                "recall_sensitivity": float(recall_score(target, pred, zero_division=0)),
                "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
                "f1": float(f1_score(target, pred, zero_division=0)),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    return rows


def aggregate_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_class = class_metric_rows(targets, probabilities, thresholds, class_names)
    predictions = (probabilities >= thresholds.reshape(1, -1)).astype(np.int8)
    aurocs = np.asarray([row["auroc"] for row in per_class], dtype=float)
    auprcs = np.asarray([row["auprc"] for row in per_class], dtype=float)
    specificities = np.asarray([row["specificity"] for row in per_class], dtype=float)
    metrics = {
        "n_records": int(len(targets)),
        "macro_auroc": safe_nanmean(aurocs),
        "macro_auprc": safe_nanmean(auprcs),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(targets, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets, predictions, average="macro", zero_division=0)),
        "macro_specificity": safe_nanmean(specificities),
        "subset_accuracy": float(np.all(targets == predictions, axis=1).mean()),
        "label_cardinality_reference": float(targets.sum(axis=1).mean()),
        "label_cardinality_prediction": float(predictions.sum(axis=1).mean()),
    }
    return metrics, per_class


def metric_snapshot(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float]:
    overall, per_class = aggregate_metrics(targets, probabilities, thresholds, class_names)
    snapshot = {
        "macro_auroc": float(overall["macro_auroc"]),
        "macro_auprc": float(overall["macro_auprc"]),
        "macro_f1": float(overall["macro_f1"]),
    }
    for row in per_class:
        name = str(row["class"])
        snapshot[f"{name}_auroc"] = float(row["auroc"])
        snapshot[f"{name}_auprc"] = float(row["auprc"])
        snapshot[f"{name}_f1"] = float(row["f1"])
    return snapshot


def patient_cluster_indices(patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    unique_patients, inverse = np.unique(patient_ids.astype(str), return_inverse=True)
    groups = [np.flatnonzero(inverse == index) for index in range(len(unique_patients))]
    draws = rng.integers(0, len(unique_patients), size=len(unique_patients))
    return np.concatenate([groups[index] for index in draws])


def bootstrap_confidence_intervals(
    *,
    targets: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    point = metric_snapshot(targets, probabilities, thresholds, class_names)
    if replicates <= 0:
        return {key: {"estimate": value, "ci95": None, "valid_replicates": 0} for key, value in point.items()}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in point}
    for _ in tqdm(range(replicates), desc="patient bootstrap", leave=False):
        indices = patient_cluster_indices(patient_ids, rng)
        values = metric_snapshot(targets[indices], probabilities[indices], thresholds, class_names)
        for key, value in values.items():
            if math.isfinite(value):
                samples[key].append(value)
    summary = {}
    for key, estimate in point.items():
        values = np.asarray(samples[key], dtype=float)
        summary[key] = {
            "estimate": estimate,
            "ci95": np.quantile(values, [0.025, 0.975]).tolist() if values.size else None,
            "valid_replicates": int(values.size),
        }
    return summary


def paired_bootstrap_difference(
    *,
    targets: np.ndarray,
    trace_probabilities: np.ndarray,
    base_probabilities: np.ndarray,
    patient_ids: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    keys = ["macro_auroc", "macro_auprc", "macro_f1"]
    trace_point = metric_snapshot(targets, trace_probabilities, thresholds, class_names)
    base_point = metric_snapshot(targets, base_probabilities, thresholds, class_names)
    point = {key: trace_point[key] - base_point[key] for key in keys}
    if replicates <= 0:
        return {key: {"estimate_trace_minus_base": value, "ci95": None, "valid_replicates": 0} for key, value in point.items()}
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {key: [] for key in keys}
    for _ in tqdm(range(replicates), desc="paired patient bootstrap", leave=False):
        indices = patient_cluster_indices(patient_ids, rng)
        trace_values = metric_snapshot(targets[indices], trace_probabilities[indices], thresholds, class_names)
        base_values = metric_snapshot(targets[indices], base_probabilities[indices], thresholds, class_names)
        for key in keys:
            value = trace_values[key] - base_values[key]
            if math.isfinite(value):
                samples[key].append(value)
    summary = {}
    for key, estimate in point.items():
        values = np.asarray(samples[key], dtype=float)
        summary[key] = {
            "estimate_trace_minus_base": estimate,
            "ci95": np.quantile(values, [0.025, 0.975]).tolist() if values.size else None,
            "valid_replicates": int(values.size),
        }
    return summary


def cohort_arrays(cohort: Cohort) -> dict[str, Any]:
    rows = cohort.selected_rows
    metadata = cohort.metadata.iloc[rows].reset_index(drop=True)
    subsets = cohort.subsets.iloc[rows].reset_index(drop=True)
    eligible = parse_bool_column(subsets["has_five_superclass_target"], "has_five_superclass_target")
    targets = cohort.targets[rows]
    mask = np.asarray(cohort.valid_mask[rows], dtype=bool)
    per_lead_fraction = mask.mean(axis=2)
    observed_fraction = per_lead_fraction.mean(axis=1)
    return {
        "rows": rows,
        "metadata": metadata,
        "subsets": subsets,
        "eligible": eligible,
        "targets": targets,
        "per_lead_fraction": per_lead_fraction,
        "observed_fraction": observed_fraction,
    }


def subgroup_rows(
    *,
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
    class_names: Sequence[str],
    layout: np.ndarray,
) -> list[dict[str, Any]]:
    normalized_layout = np.asarray(
        ["PSCRIPT" if str(value) == "PSCRIPT_BUNDLE" else str(value) for value in layout]
    )
    groups: list[tuple[str, np.ndarray]] = [("ALL", np.ones(len(targets), dtype=bool))]
    groups.extend((name, normalized_layout == name) for name in sorted(set(normalized_layout)))
    rows: list[dict[str, Any]] = []
    for group_name, mask in groups:
        metrics, per_class = aggregate_metrics(targets[mask], probabilities[mask], thresholds, class_names)
        row = {"subgroup_type": "layout", "subgroup": group_name, **metrics}
        for item in per_class:
            name = item["class"]
            row[f"{name}_support"] = item["support_positive"]
            row[f"{name}_auroc"] = item["auroc"]
            row[f"{name}_auprc"] = item["auprc"]
            row[f"{name}_f1"] = item["f1"]
        rows.append(row)
    return rows


def prediction_table(
    *,
    arrays: Mapping[str, Any],
    class_names: Sequence[str],
    probabilities: np.ndarray,
    threshold_schemes: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    metadata = arrays["metadata"]
    subsets = arrays["subsets"]
    eligible = arrays["eligible"]
    targets = arrays["targets"]
    columns = [
        "ecg_id",
        "record_uid",
        "patient_id",
        "age",
        "sex",
        "source_template",
        "layout_subtype",
        "recommended_split",
    ]
    table = metadata[columns].copy()
    table.insert(0, "row_index", arrays["rows"])
    table["has_five_superclass_target"] = eligible
    table["diagnostic_superclasses"] = subsets["diagnostic_superclasses"]
    table["observed_fraction"] = arrays["observed_fraction"]
    for index, lead in enumerate(LEADS):
        table[f"observed_fraction_{lead}"] = arrays["per_lead_fraction"][:, index]
    for index, class_name in enumerate(class_names):
        table[f"target_{class_name}"] = np.where(eligible, targets[:, index], -1)
        table[f"prob_{class_name}"] = probabilities[:, index]
        for scheme_name, threshold in threshold_schemes.items():
            table[f"pred_{scheme_name}_{class_name}"] = (
                probabilities[:, index] >= float(threshold[index])
            ).astype(np.int8)
    return table


def verify_output_target(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinels = [output_dir / "protocol.json", output_dir / "comparison_trace_vs_base.json"]
    if not overwrite and any(path.exists() for path in sentinels):
        raise FileExistsError(
            f"Output already exists at {output_dir}; use a new directory or pass --overwrite"
        )


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.bootstrap_seed)
    np.random.seed(args.bootstrap_seed)
    torch.manual_seed(args.bootstrap_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.bootstrap_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    verify_output_target(args.output_dir, args.overwrite)
    cohort = load_cohort(args)
    arrays = cohort_arrays(cohort)
    threshold_schemes, threshold_provenance = load_threshold_schemes(
        args.threshold_config, cohort.class_names
    )
    dataset = ExternalInferenceDataset(cohort.signals, cohort.selected_rows)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.disable_amp
    eligible = arrays["eligible"]
    if not bool(eligible.any()):
        raise ValueError("Selected rows contain no five-superclass evaluation records")
    eval_targets = arrays["targets"][eligible]
    eval_patient_ids = arrays["metadata"].loc[eligible, "patient_id"].astype(str).to_numpy()
    eval_layout = arrays["metadata"].loc[eligible, "layout_subtype"].astype(str).to_numpy()

    checkpoints = {"base": args.base_checkpoint, "trace": args.trace_checkpoint}
    model_probabilities: dict[str, np.ndarray] = {}
    model_summaries: dict[str, Any] = {}
    for kind in args.models:
        print(f"Loading locked {kind} checkpoint: {checkpoints[kind]}", flush=True)
        model, checkpoint_info = checkpoint_model(
            kind=kind,
            checkpoint_path=checkpoints[kind],
            tokenizer_checkpoint=args.tokenizer_checkpoint,
            expected_class_names=cohort.class_names,
            device=device,
        )
        logits, inferred_rows = collect_logits(
            model,
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            use_amp=use_amp,
        )
        if not np.array_equal(inferred_rows, cohort.selected_rows):
            raise RuntimeError("Inference row order changed unexpectedly")
        probabilities = sigmoid(logits)
        model_probabilities[kind] = probabilities
        eval_probabilities = probabilities[eligible]
        model_dir = args.output_dir / kind
        model_dir.mkdir(parents=True, exist_ok=True)

        scheme_metrics: dict[str, Any] = {}
        per_class_tables = []
        for scheme_name, thresholds in threshold_schemes.items():
            metrics, per_class = aggregate_metrics(
                eval_targets, eval_probabilities, thresholds, cohort.class_names
            )
            scheme_metrics[scheme_name] = metrics
            per_class_frame = pd.DataFrame(per_class)
            per_class_frame.insert(0, "threshold_scheme", scheme_name)
            per_class_tables.append(per_class_frame)

        primary_thresholds = threshold_schemes["fixed_0p5_primary"]
        bootstrap = bootstrap_confidence_intervals(
            targets=eval_targets,
            probabilities=eval_probabilities,
            patient_ids=eval_patient_ids,
            thresholds=primary_thresholds,
            class_names=cohort.class_names,
            replicates=args.bootstrap_replicates,
            seed=args.bootstrap_seed,
        )
        subgroups = subgroup_rows(
            targets=eval_targets,
            probabilities=eval_probabilities,
            thresholds=primary_thresholds,
            class_names=cohort.class_names,
            layout=eval_layout,
        )
        predictions = prediction_table(
            arrays=arrays,
            class_names=cohort.class_names,
            probabilities=probabilities,
            threshold_schemes=threshold_schemes,
        )
        predictions.to_csv(model_dir / "predictions.csv", index=False)
        pd.concat(per_class_tables, ignore_index=True).to_csv(
            model_dir / "per_class_metrics.csv", index=False
        )
        pd.DataFrame(subgroups).to_csv(model_dir / "subgroup_metrics.csv", index=False)
        np.savez_compressed(
            model_dir / "predictions.npz",
            row_index=arrays["rows"],
            logits=logits.astype(np.float32),
            probabilities=probabilities.astype(np.float32),
            targets=np.where(eligible[:, None], arrays["targets"], -1).astype(np.int8),
            eligible=eligible,
            observed_fraction=arrays["observed_fraction"].astype(np.float32),
            class_names=np.asarray(cohort.class_names),
        )
        summary = {
            "protocol": "locked PTB-XL checkpoint; zero Anzhen training/validation/model selection",
            "checkpoint": checkpoint_info,
            "n_inferred_records": int(len(arrays["rows"])),
            "n_evaluable_records": int(eligible.sum()),
            "n_evaluable_patients": int(len(np.unique(eval_patient_ids))),
            "thresholds": threshold_provenance,
            "metrics": scheme_metrics,
            "primary_fixed_0p5_patient_cluster_bootstrap": bootstrap,
        }
        write_json(model_dir / "metrics.json", summary)
        model_summaries[kind] = summary
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    comparison = None
    if {"base", "trace"}.issubset(model_probabilities):
        primary_thresholds = threshold_schemes["fixed_0p5_primary"]
        comparison = {
            "contrast": "TRACE minus Base Branch on identical Anzhen records",
            "threshold_scheme": "fixed_0p5_primary",
            "paired_patient_cluster_bootstrap": paired_bootstrap_difference(
                targets=eval_targets,
                trace_probabilities=model_probabilities["trace"][eligible],
                base_probabilities=model_probabilities["base"][eligible],
                patient_ids=eval_patient_ids,
                thresholds=primary_thresholds,
                class_names=cohort.class_names,
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            ),
        }
        write_json(args.output_dir / "comparison_trace_vs_base.json", comparison)

    artifact_paths = {
        "full_data": args.data_root / "trace_compat/full_data.npy",
        "valid_mask": args.data_root / "trace_compat/valid_mask.npy",
        "metadata": args.data_root / "trace_compat/ptbxl_compat_database.csv",
        "subsets": args.data_root / "trace_compat/evaluation_subsets.csv",
        "trace_checkpoint": resolve_repo_path(args.trace_checkpoint),
        "base_checkpoint": resolve_repo_path(args.base_checkpoint),
        "tokenizer_checkpoint": resolve_repo_path(args.tokenizer_checkpoint),
        "threshold_config": resolve_repo_path(args.threshold_config),
        "evaluation_code": Path(__file__).resolve(),
    }
    hashes = {}
    if not args.skip_integrity_hashes:
        for name, path in tqdm(artifact_paths.items(), desc="sha256", leave=False):
            hashes[name] = sha256_file(path)

    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Anzhen locked zero-shot external diagnosis evaluation",
        "data_role": {
            "training_records": 0,
            "validation_records": 0,
            "model_selection_records": 0,
            "threshold_selection_records": 0,
            "inference_records": int(len(arrays["rows"])),
            "five_superclass_evaluation_records": int(eligible.sum()),
            "five_superclass_evaluation_patients": int(len(np.unique(eval_patient_ids))),
        },
        "source_representation": "de-identified PTB-XL-compatible 12-lead ECG representation",
        "trace_input": {
            "stored_shape": list(np.load(args.data_root / "trace_compat/full_data.npy", mmap_mode="r").shape),
            "evaluated_window_shape": [int(len(arrays["rows"])), 12, args.seq_length],
            "periodic_completion_warning": "Compatibility filling does not restore unobserved physiology or true 12-lead synchrony.",
        },
        "label_reference": "clinician-verified source cases and diagnoses with a rule-based PTB-XL/SCP compatibility mapping",
        "class_names": cohort.class_names,
        "models": list(args.models),
        "thresholds": threshold_provenance,
        "bootstrap": {
            "unit": "patient_id cluster",
            "replicates": args.bootstrap_replicates,
            "seed": args.bootstrap_seed,
        },
        "artifacts": {name: str(path) for name, path in artifact_paths.items()},
        "sha256": hashes,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "device": str(device),
            "amp": use_amp,
            "elapsed_seconds": time.time() - started,
        },
        "smoke_test_max_samples": args.max_samples,
        "comparison_available": comparison is not None,
    }
    write_json(args.output_dir / "protocol.json", protocol)
    print(json.dumps(json_safe({
        "output_dir": str(args.output_dir),
        "data_role": protocol["data_role"],
        "primary_metrics": {
            model: summary["metrics"]["fixed_0p5_primary"]
            for model, summary in model_summaries.items()
        },
        "comparison": comparison,
        "elapsed_seconds": protocol["runtime"]["elapsed_seconds"],
    }), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
