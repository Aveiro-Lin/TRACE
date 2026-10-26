#!/usr/bin/env python3
"""Patient-level Anzhen fine-tuning from a PTB-XL TRACE checkpoint.

The 50:50 protocol reserves half of the eligible patients as a locked test set.
The development half is split 80:20 into training and validation patients,
yielding an overall 40:10:50 train/validation/test allocation.  The same entry
point also supports a later 70:15:15 experiment without changing the protocol.
"""

from __future__ import annotations

import argparse
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluate_anzhen_external import (
    DEFAULT_DATA_ROOT,
    aggregate_metrics,
    bootstrap_confidence_intervals,
    json_safe,
    load_cohort,
    parse_bool_column,
    sha256_file,
    sigmoid,
    subgroup_rows,
    write_json,
)
from utils.checkpoint_compat import remap_legacy_foundation_encoder_keys
from utils.downstream_trace import TRACEDiagnosisClassifier


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_CHECKPOINT = REPO_ROOT / "ckpt/ptbxl/downstream_refined/best.pt"
DEFAULT_TOKENIZER_CHECKPOINT = (
    REPO_ROOT
    / "ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results/anzhen_diagnosis"
SPLIT_RATIOS = {
    "50_50": {"train": 0.40, "valid": 0.10, "test": 0.50},
    "70_15_15": {"train": 0.70, "valid": 0.15, "test": 0.15},
}


@dataclass(frozen=True)
class SplitArtifacts:
    manifest: pd.DataFrame
    indices: dict[str, np.ndarray]
    summary: dict[str, Any]


class AnzhenFineTuneDataset(Dataset):
    def __init__(self, signals: np.ndarray, targets: np.ndarray, indices: np.ndarray):
        self.signals = signals
        self.targets = np.asarray(targets, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        row = int(self.indices[index])
        # The source is a read-only mmap.  Copy so PyTorch never receives a
        # non-writable array and worker processes cannot mutate source data.
        signal = np.array(self.signals[row], dtype=np.float32, copy=True)
        target = np.array(self.targets[row], dtype=np.float32, copy=True)
        return torch.from_numpy(signal), torch.from_numpy(target), row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-level fine-tuning of PTB-XL TRACE on Anzhen ECGs"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=Path,
        default=DEFAULT_TOKENIZER_CHECKPOINT,
        help="Tokenizer used to reconstruct the diagnosis model before strict checkpoint loading.",
    )
    parser.add_argument("--split-protocol", choices=tuple(SPLIT_RATIOS), default="50_50")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-trials", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight-cap", type=float, default=10.0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "heads", "residual"),
        default="all",
        help="all uses discriminative learning rates; heads freezes the encoder; residual trains only the TRACE residual branch.",
    )
    parser.add_argument("--target-metric", choices=("macro_auroc", "macro_auprc"), default="macro_auroc")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seq-length", type=int, default=2250)
    parser.add_argument("--latent-ratio", type=float, default=0.1)
    # Kept for API compatibility with evaluate_anzhen_external.load_cohort;
    # fine-tuning always uses the complete eligible cohort.
    parser.set_defaults(max_samples=None)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test only.")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Smoke-test only.")
    parser.add_argument("--split-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
    value = torch.initial_seed() % (2**32)
    np.random.seed(value)
    random.seed(value)


def exact_split_sizes(n_patients: int, ratios: Mapping[str, float]) -> dict[str, int]:
    names = list(ratios)
    raw = np.asarray([float(ratios[name]) * n_patients for name in names])
    counts = np.floor(raw).astype(int)
    remainder = n_patients - int(counts.sum())
    # Stable ordering makes tied fractional remainders reproducible.
    order = np.argsort(-(raw - counts), kind="stable")
    for index in order[:remainder]:
        counts[index] += 1
    return {name: int(count) for name, count in zip(names, counts)}


def iterative_multilabel_assignment(
    features: np.ndarray,
    split_names: Sequence[str],
    split_sizes: Mapping[str, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Greedy iterative stratification with exact patient capacities."""
    n_patients, n_features = features.shape
    capacities = np.asarray([split_sizes[name] for name in split_names], dtype=int)
    ratios = capacities / float(n_patients)
    desired = ratios[:, None] * features.sum(axis=0, keepdims=True)
    remaining_need = desired.copy()
    remaining_capacity = capacities.copy()
    assignment = np.full(n_patients, -1, dtype=np.int16)

    while bool(np.any(assignment < 0)):
        unassigned = np.flatnonzero(assignment < 0)
        remaining_positive = features[unassigned].sum(axis=0)
        available_features = np.flatnonzero(remaining_positive > 0)
        if available_features.size == 0:
            shuffled = rng.permutation(unassigned)
            for patient in shuffled:
                choices = np.flatnonzero(remaining_capacity > 0)
                max_capacity = remaining_capacity[choices].max()
                choices = choices[remaining_capacity[choices] == max_capacity]
                chosen = int(rng.choice(choices))
                assignment[patient] = chosen
                remaining_capacity[chosen] -= 1
            break

        rare_count = remaining_positive[available_features].min()
        rare_features = available_features[remaining_positive[available_features] == rare_count]
        feature_index = int(rng.choice(rare_features))
        candidates = unassigned[features[unassigned, feature_index] > 0]
        candidates = rng.permutation(candidates)

        for patient in candidates:
            if assignment[patient] >= 0:
                continue
            choices = np.flatnonzero(remaining_capacity > 0)
            patient_features = np.flatnonzero(features[patient] > 0)
            if patient_features.size:
                denominators = np.maximum(desired[:, patient_features], 1.0)
                feature_need = (
                    np.maximum(remaining_need[:, patient_features], 0.0) / denominators
                ).mean(axis=1)
            else:
                feature_need = np.zeros(len(split_names), dtype=float)
            capacity_need = remaining_capacity / np.maximum(capacities, 1)
            score = feature_need + 0.05 * capacity_need
            best = score[choices].max()
            tied = choices[np.isclose(score[choices], best)]
            if tied.size > 1:
                max_capacity = remaining_capacity[tied].max()
                tied = tied[remaining_capacity[tied] == max_capacity]
            chosen = int(rng.choice(tied))
            assignment[patient] = chosen
            remaining_capacity[chosen] -= 1
            remaining_need[chosen] -= features[patient]

    if not np.array_equal(
        np.bincount(assignment, minlength=len(split_names)),
        capacities,
    ):
        raise RuntimeError("Iterative stratifier failed to meet exact patient capacities")
    return assignment


def assignment_score(
    assignment: np.ndarray,
    features: np.ndarray,
    label_count: int,
    split_names: Sequence[str],
    split_sizes: Mapping[str, int],
    patient_record_counts: np.ndarray,
) -> tuple[float, float, float, float]:
    capacities = np.asarray([split_sizes[name] for name in split_names], dtype=float)
    ratios = capacities / capacities.sum()
    actual = np.stack([features[assignment == index].sum(axis=0) for index in range(len(split_names))])
    expected = ratios[:, None] * features.sum(axis=0, keepdims=True)
    scaled_error = np.abs(actual - expected) / np.maximum(expected, 1.0)
    label_error = scaled_error[:, :label_count]
    layout_error = scaled_error[:, label_count:]
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


def build_patient_level_split(cohort: Any, protocol: str, seed: int, trials: int) -> SplitArtifacts:
    eligible = parse_bool_column(
        cohort.subsets["has_five_superclass_target"], "has_five_superclass_target"
    )
    eligible_rows = np.flatnonzero(eligible)
    metadata = cohort.metadata.iloc[eligible_rows].copy()
    metadata["row_index"] = eligible_rows
    metadata["patient_id"] = metadata["patient_id"].astype(str)
    targets = cohort.targets[eligible_rows].astype(np.int8)

    layout = metadata["layout_subtype"].astype(str).replace({"PSCRIPT_BUNDLE": "PSCRIPT"})
    layout_names = sorted(layout.unique().tolist())
    row_frame = pd.DataFrame(targets, columns=cohort.class_names, index=metadata.index)
    row_frame["patient_id"] = metadata["patient_id"].to_numpy()
    for name in layout_names:
        row_frame[f"layout::{name}"] = (layout.to_numpy() == name).astype(np.int8)
    row_frame["record_count"] = 1

    feature_names = list(cohort.class_names) + [f"layout::{name}" for name in layout_names]
    patient_frame = row_frame.groupby("patient_id", sort=True)[feature_names].max()
    record_counts = row_frame.groupby("patient_id", sort=True)["record_count"].sum()
    patient_ids = patient_frame.index.to_numpy(dtype=str)
    features = patient_frame.to_numpy(dtype=np.int8)
    patient_record_counts = record_counts.to_numpy(dtype=np.int64)
    split_names = ["train", "valid", "test"]
    split_sizes = exact_split_sizes(len(patient_ids), SPLIT_RATIOS[protocol])

    if trials <= 0:
        raise ValueError("--split-trials must be positive")
    best_assignment: np.ndarray | None = None
    best_score: tuple[float, float, float, float] | None = None
    best_trial = -1
    for trial in range(trials):
        rng = np.random.default_rng(seed + 104729 * trial)
        assignment = iterative_multilabel_assignment(
            features, split_names, split_sizes, rng
        )
        score = assignment_score(
            assignment,
            features,
            len(cohort.class_names),
            split_names,
            split_sizes,
            patient_record_counts,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment.copy()
            best_trial = trial

    assert best_assignment is not None and best_score is not None
    patient_to_split = {
        patient: split_names[int(split_index)]
        for patient, split_index in zip(patient_ids, best_assignment)
    }
    split_values = metadata["patient_id"].map(patient_to_split)
    if split_values.isna().any():
        raise RuntimeError("Some eligible patients were not assigned to a split")

    manifest = metadata[
        [
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
    ].copy()
    manifest["split"] = split_values.to_numpy()
    manifest["development_half"] = manifest["split"].isin(["train", "valid"])
    manifest["diagnostic_superclasses"] = cohort.subsets.iloc[eligible_rows][
        "diagnostic_superclasses"
    ].to_numpy()
    for class_index, class_name in enumerate(cohort.class_names):
        manifest[f"target_{class_name}"] = targets[:, class_index]
    manifest = manifest.sort_values("row_index").reset_index(drop=True)

    indices: dict[str, np.ndarray] = {}
    for split_name in split_names:
        indices[split_name] = manifest.loc[
            manifest["split"].eq(split_name), "row_index"
        ].to_numpy(dtype=np.int64)

    patient_sets = {
        name: set(manifest.loc[manifest["split"].eq(name), "patient_id"].astype(str))
        for name in split_names
    }
    overlaps = {
        "train_valid": len(patient_sets["train"] & patient_sets["valid"]),
        "train_test": len(patient_sets["train"] & patient_sets["test"]),
        "valid_test": len(patient_sets["valid"] & patient_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Patient leakage detected: {overlaps}")

    split_rows = []
    for split_name in split_names:
        part = manifest[manifest["split"].eq(split_name)]
        item: dict[str, Any] = {
            "split": split_name,
            "patients": int(part["patient_id"].nunique()),
            "records": int(len(part)),
        }
        for class_name in cohort.class_names:
            item[f"{class_name}_positive_records"] = int(part[f"target_{class_name}"].sum())
            item[f"{class_name}_prevalence"] = float(part[f"target_{class_name}"].mean())
        normalized_layout = part["layout_subtype"].replace({"PSCRIPT_BUNDLE": "PSCRIPT"})
        for layout_name in layout_names:
            item[f"layout_{layout_name}_records"] = int(normalized_layout.eq(layout_name).sum())
        split_rows.append(item)

    summary = {
        "protocol": protocol,
        "protocol_interpretation": (
            "50% development (80% train, 20% validation) and 50% locked test; "
            "overall 40:10:50"
            if protocol == "50_50"
            else "70% train, 15% validation, and 15% locked test"
        ),
        "unit_of_split": "patient_id",
        "eligible_definition": "has_five_superclass_target == True",
        "eligible_records": int(len(manifest)),
        "eligible_patients": int(len(patient_ids)),
        "excluded_records_without_five_superclass_target": int(len(cohort.metadata) - len(manifest)),
        "requested_ratios": SPLIT_RATIOS[protocol],
        "exact_patient_counts": split_sizes,
        "stratification_features": feature_names,
        "stratification_trials": int(trials),
        "selected_trial": int(best_trial),
        "balance_objective": list(best_score),
        "patient_overlap": overlaps,
        "splits": split_rows,
    }
    return SplitArtifacts(manifest=manifest, indices=indices, summary=summary)


def build_model(
    source_checkpoint: Path,
    tokenizer_checkpoint: Path,
    class_names: Sequence[str],
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    source_checkpoint = resolve_repo_path(source_checkpoint)
    checkpoint = torch.load(source_checkpoint, map_location="cpu")
    saved_classes = list(checkpoint.get("class_names", []))
    if saved_classes != list(class_names):
        raise ValueError(f"Checkpoint classes {saved_classes} != Anzhen classes {list(class_names)}")
    saved_args = dict(checkpoint.get("args", {}))
    tokenizer_checkpoint = resolve_repo_path(tokenizer_checkpoint)
    if not tokenizer_checkpoint.exists():
        raise FileNotFoundError(f"Tokenizer checkpoint not found: {tokenizer_checkpoint}")
    model = TRACEDiagnosisClassifier(
        tokenizer_checkpoint=str(tokenizer_checkpoint),
        num_classes=len(class_names),
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
        base_downstream_checkpoint="",
        freeze_base_branch=False,
    )
    model_state, remapped_keys = remap_legacy_foundation_encoder_keys(
        checkpoint["model_state_dict"]
    )
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    provenance = {
        "checkpoint": str(source_checkpoint),
        "sha256": sha256_file(source_checkpoint),
        "epoch": checkpoint.get("epoch"),
        "validation_metrics": checkpoint.get("metrics", {}),
        "saved_args": saved_args,
        "tokenizer_checkpoint": str(tokenizer_checkpoint),
        "legacy_foundation_encoder_keys_remapped": remapped_keys,
        "initialization": "strict full-state load; optimizer state deliberately not reused",
    }
    return model, provenance


def configure_trainable_scope(model: nn.Module, scope: str) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.freeze_backbone = scope != "all"
    if scope == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    elif scope == "heads":
        for name, parameter in model.named_parameters():
            if not name.startswith("backbone."):
                parameter.requires_grad_(True)
    elif scope == "residual":
        prefixes = ("delta_query", "residual_scale", "delta_token_norm.", "delta_hidden.", "delta_head.")
        for name, parameter in model.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad_(True)
    else:
        raise ValueError(f"Unknown trainable scope: {scope}")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("No trainable parameters were selected")
    return {"total_parameters": int(total), "trainable_parameters": int(trainable)}


def build_optimizer(model: nn.Module, args: argparse.Namespace) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    backbone = []
    heads = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("backbone.") else heads).append(parameter)
    groups = []
    summary = []
    if backbone:
        groups.append({"params": backbone, "lr": args.backbone_lr})
        summary.append({"name": "backbone", "lr": args.backbone_lr, "parameters": sum(p.numel() for p in backbone)})
    if heads:
        groups.append({"params": heads, "lr": args.head_lr})
        summary.append({"name": "heads_and_residual", "lr": args.head_lr, "parameters": sum(p.numel() for p in heads)})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    return optimizer, summary


def make_loaders(
    cohort: Any,
    indices: Mapping[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, DataLoader]:
    datasets = {
        name: AnzhenFineTuneDataset(cohort.signals, cohort.targets, split_indices)
        for name, split_indices in indices.items()
    }
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_seed,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "valid": DataLoader(datasets["valid"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


def training_pos_weight(targets: np.ndarray, train_indices: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray]:
    train_targets = targets[train_indices].astype(np.float64)
    positive = train_targets.sum(axis=0)
    negative = len(train_targets) - positive
    if np.any(positive <= 0):
        raise ValueError(f"Training split has a class with no positives: {positive.tolist()}")
    raw = negative / positive
    effective = np.minimum(raw, cap) if cap > 0 else raw
    return raw.astype(np.float32), effective.astype(np.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
    gradient_clip: float,
    max_batches: int | None,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0
    for batch_index, (signals, targets, _) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        signals = signals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(signals)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward()
        if gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                gradient_clip,
            )
        scaler.step(optimizer)
        scaler.update()
        batch_size = int(signals.size(0))
        loss_sum += float(loss.detach().cpu()) * batch_size
        count += batch_size
    return loss_sum / max(count, 1)


@torch.inference_mode()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    class_names: Sequence[str],
    max_batches: int | None = None,
) -> dict[str, Any]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    loss_sum = 0.0
    count = 0
    logits_buffer = []
    targets_buffer = []
    rows_buffer = []
    for batch_index, (signals, targets, rows) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        signals = signals.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(signals)
            loss = loss_fn(logits, targets_device)
        batch_size = int(signals.size(0))
        loss_sum += float(loss.detach().cpu()) * batch_size
        count += batch_size
        logits_buffer.append(logits.float().cpu())
        targets_buffer.append(targets.float().cpu())
        rows_buffer.append(rows.long().cpu())
    if not logits_buffer:
        raise RuntimeError("Evaluation produced no batches")
    logits = torch.cat(logits_buffer).numpy()
    targets = torch.cat(targets_buffer).numpy().astype(np.int8)
    rows = torch.cat(rows_buffer).numpy().astype(np.int64)
    probabilities = sigmoid(logits)
    fixed_thresholds = np.full(len(class_names), 0.5, dtype=np.float32)
    metrics, per_class = aggregate_metrics(targets, probabilities, fixed_thresholds, class_names)
    return {
        "loss": loss_sum / max(count, 1),
        "logits": logits,
        "targets": targets,
        "probabilities": probabilities,
        "rows": rows,
        "metrics_fixed_0p5": metrics,
        "per_class_fixed_0p5": per_class,
    }


def validation_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.01, 0.99, 99, dtype=np.float32)
    thresholds = []
    for class_index in range(targets.shape[1]):
        target = targets[:, class_index]
        scores = []
        for threshold in grid:
            prediction = probabilities[:, class_index] >= threshold
            tp = float(np.sum((target == 1) & prediction))
            fp = float(np.sum((target == 0) & prediction))
            fn = float(np.sum((target == 1) & ~prediction))
            denominator = 2 * tp + fp + fn
            scores.append(2 * tp / denominator if denominator else 0.0)
        scores = np.asarray(scores)
        best = np.flatnonzero(np.isclose(scores, scores.max()))
        closest = best[np.argmin(np.abs(grid[best] - 0.5))]
        thresholds.append(float(grid[closest]))
    return np.asarray(thresholds, dtype=np.float32)


def selection_metric(result: Mapping[str, Any], name: str) -> float:
    if name == "macro_auroc":
        return float(result["metrics_fixed_0p5"]["macro_auroc"])
    values = []
    for index in range(result["targets"].shape[1]):
        target = result["targets"][:, index]
        if np.unique(target).size == 2:
            values.append(average_precision_score(target, result["probabilities"][:, index]))
    return float(np.mean(values)) if values else float("nan")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: Mapping[str, Any],
    class_names: Sequence[str],
    args: argparse.Namespace,
    source_checkpoint: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": dict(metrics),
            "class_names": list(class_names),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "source_checkpoint": str(source_checkpoint),
        },
        path,
    )


def load_model_state(path: Path, model: nn.Module) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    model_state, _ = remap_legacy_foundation_encoder_keys(checkpoint["model_state_dict"])
    model.load_state_dict(model_state, strict=True)
    return checkpoint


def prediction_frame(
    manifest: pd.DataFrame,
    result: Mapping[str, Any],
    class_names: Sequence[str],
    thresholds: np.ndarray,
    model_stage: str,
) -> pd.DataFrame:
    by_row = manifest.set_index("row_index")
    frame = by_row.loc[result["rows"]].reset_index()[
        ["row_index", "ecg_id", "record_uid", "patient_id", "split", "layout_subtype"]
    ].copy()
    frame.insert(0, "model_stage", model_stage)
    for index, class_name in enumerate(class_names):
        frame[f"target_{class_name}"] = result["targets"][:, index]
        frame[f"prob_{class_name}"] = result["probabilities"][:, index]
        frame[f"pred_fixed_0p5_{class_name}"] = (
            result["probabilities"][:, index] >= 0.5
        ).astype(np.int8)
        frame[f"pred_val_threshold_{class_name}"] = (
            result["probabilities"][:, index] >= thresholds[index]
        ).astype(np.int8)
    return frame


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    started = time.time()
    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_ROOT / f"split_{args.split_protocol}_seed{args.seed}"
    output_dir = resolve_repo_path(args.output_dir)
    source_checkpoint = resolve_repo_path(args.source_checkpoint)
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; choose a new path or pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    cohort = load_cohort(args)
    split = build_patient_level_split(
        cohort, args.split_protocol, args.seed, args.split_trials
    )
    split.manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    write_json(output_dir / "split_summary.json", split.summary)
    pd.DataFrame(split.summary["splits"]).to_csv(output_dir / "split_distribution.csv", index=False)
    excluded = cohort.subsets.loc[
        ~parse_bool_column(cohort.subsets["has_five_superclass_target"], "has_five_superclass_target"),
        ["row_index", "ecg_id", "record_uid", "diagnostic_superclasses"],
    ].copy()
    excluded.insert(3, "patient_id", cohort.metadata.loc[excluded.index, "patient_id"].astype(str).to_numpy())
    excluded["exclusion_reason"] = "no_PTBXL_five_superclass_target"
    excluded.to_csv(output_dir / "excluded_records.csv", index=False)
    print(json.dumps(json_safe(split.summary), ensure_ascii=False, indent=2), flush=True)
    if args.split_only:
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    use_amp = device.type == "cuda" and not args.disable_amp
    loaders = make_loaders(cohort, split.indices, args, device)
    model, source_provenance = build_model(
        source_checkpoint,
        args.tokenizer_checkpoint,
        cohort.class_names,
        device,
    )
    parameter_summary = configure_trainable_scope(model, args.trainable_scope)
    optimizer, optimizer_summary = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    raw_pos_weight, effective_pos_weight = training_pos_weight(
        cohort.targets, split.indices["train"], args.pos_weight_cap
    )
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(effective_pos_weight, dtype=torch.float32, device=device)
    )
    scaler = GradScaler(enabled=use_amp)

    protocol = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Anzhen PTB-XL five-superclass domain fine-tuning",
        "split": split.summary,
        "source": source_provenance,
        "fine_tuning": {
            "trainable_scope": args.trainable_scope,
            **parameter_summary,
            "optimizer": "AdamW with a fresh optimizer state",
            "optimizer_groups": optimizer_summary,
            "weight_decay": args.weight_decay,
            "training_loss": "class-weighted BCEWithLogitsLoss",
            "raw_pos_weight": dict(zip(cohort.class_names, raw_pos_weight.tolist())),
            "effective_pos_weight": dict(zip(cohort.class_names, effective_pos_weight.tolist())),
            "pos_weight_cap": args.pos_weight_cap,
            "checkpoint_selection": f"validation {args.target_metric}",
            "test_use_during_training_or_selection": "none",
            "primary_threshold": 0.5,
            "secondary_thresholds": "per-class F1 thresholds selected on validation only",
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "minimum_epochs": args.min_epochs,
            "seed": args.seed,
            "amp": use_amp,
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
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
            criterion,
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
            cohort.class_names,
            args.max_eval_batches,
        )
        metric_value = selection_metric(valid_result, args.target_metric)
        row = {
            "epoch": epoch,
            "train_loss_weighted": train_loss,
            "valid_loss_unweighted": valid_result["loss"],
            "selection_metric": args.target_metric,
            "selection_value": metric_value,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            **valid_result["metrics_fixed_0p5"],
        }
        if hasattr(model, "booster_debug_stats"):
            row.update(model.booster_debug_stats())
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
            cohort.class_names,
            args,
            source_checkpoint,
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
                cohort.class_names,
                args,
                source_checkpoint,
            )
        else:
            stale_epochs += 1
        scheduler.step()
        if epoch + 1 >= args.min_epochs and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}", flush=True)
            break

    # The locked test set is first accessed after all training and model
    # selection are complete.  Both the source and fine-tuned checkpoints are
    # evaluated now so their comparison uses exactly the same test patients.
    source_checkpoint_payload = torch.load(source_checkpoint, map_location="cpu")
    source_model_state, _ = remap_legacy_foundation_encoder_keys(
        source_checkpoint_payload["model_state_dict"]
    )
    model.load_state_dict(source_model_state, strict=True)
    initial_test = evaluate_loader(model, loaders["test"], device, use_amp, cohort.class_names)

    best_checkpoint = load_model_state(output_dir / "best.pt", model)
    final_valid = evaluate_loader(model, loaders["valid"], device, use_amp, cohort.class_names)
    selected_thresholds = validation_thresholds(
        final_valid["targets"], final_valid["probabilities"]
    )
    final_test = evaluate_loader(model, loaders["test"], device, use_amp, cohort.class_names)
    fixed_thresholds = np.full(len(cohort.class_names), 0.5, dtype=np.float32)
    initial_metrics, initial_per_class = aggregate_metrics(
        initial_test["targets"], initial_test["probabilities"], fixed_thresholds, cohort.class_names
    )
    final_fixed_metrics, final_fixed_per_class = aggregate_metrics(
        final_test["targets"], final_test["probabilities"], fixed_thresholds, cohort.class_names
    )
    final_selected_metrics, final_selected_per_class = aggregate_metrics(
        final_test["targets"], final_test["probabilities"], selected_thresholds, cohort.class_names
    )
    test_patient_ids = (
        split.manifest.set_index("row_index").loc[final_test["rows"], "patient_id"].astype(str).to_numpy()
    )
    bootstrap_fixed = bootstrap_confidence_intervals(
        targets=final_test["targets"],
        probabilities=final_test["probabilities"],
        patient_ids=test_patient_ids,
        thresholds=fixed_thresholds,
        class_names=cohort.class_names,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    bootstrap_selected = bootstrap_confidence_intervals(
        targets=final_test["targets"],
        probabilities=final_test["probabilities"],
        patient_ids=test_patient_ids,
        thresholds=selected_thresholds,
        class_names=cohort.class_names,
        replicates=args.bootstrap_replicates,
        seed=args.seed + 1,
    )

    test_layout = (
        split.manifest.set_index("row_index").loc[final_test["rows"], "layout_subtype"].astype(str).to_numpy()
    )
    subgroup_fixed = subgroup_rows(
        targets=final_test["targets"],
        probabilities=final_test["probabilities"],
        thresholds=fixed_thresholds,
        class_names=cohort.class_names,
        layout=test_layout,
    )
    pd.DataFrame(subgroup_fixed).to_csv(output_dir / "test_subgroup_metrics_fixed_0p5.csv", index=False)

    initial_table = pd.DataFrame(initial_per_class)
    initial_table.insert(0, "model_stage", "source_ptbxl_zero_shot")
    initial_table.insert(1, "threshold_scheme", "fixed_0p5")
    fixed_table = pd.DataFrame(final_fixed_per_class)
    fixed_table.insert(0, "model_stage", "anzhen_finetuned")
    fixed_table.insert(1, "threshold_scheme", "fixed_0p5")
    selected_table = pd.DataFrame(final_selected_per_class)
    selected_table.insert(0, "model_stage", "anzhen_finetuned")
    selected_table.insert(1, "threshold_scheme", "validation_selected_per_class")
    pd.concat([initial_table, fixed_table, selected_table], ignore_index=True).to_csv(
        output_dir / "test_per_class_metrics.csv", index=False
    )

    predictions = pd.concat(
        [
            prediction_frame(split.manifest, initial_test, cohort.class_names, selected_thresholds, "source_ptbxl_zero_shot"),
            prediction_frame(split.manifest, final_test, cohort.class_names, selected_thresholds, "anzhen_finetuned"),
        ],
        ignore_index=True,
    )
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        row_index=final_test["rows"],
        targets=final_test["targets"],
        source_probabilities=initial_test["probabilities"].astype(np.float32),
        finetuned_probabilities=final_test["probabilities"].astype(np.float32),
        validation_selected_thresholds=selected_thresholds,
        class_names=np.asarray(cohort.class_names),
    )

    delta = {
        key: float(final_fixed_metrics[key] - initial_metrics[key])
        for key in ("macro_auroc", "macro_auprc", "macro_f1")
    }
    summary = {
        "protocol": split.summary["protocol_interpretation"],
        "source_checkpoint": source_provenance,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_metrics": best_checkpoint.get("metrics", {}),
        "test_access": "only after completion of training and validation-based checkpoint selection",
        "test_patients": int(len(np.unique(test_patient_ids))),
        "test_records": int(len(final_test["targets"])),
        "validation_selected_thresholds": dict(zip(cohort.class_names, selected_thresholds.tolist())),
        "source_zero_shot_test_fixed_0p5": initial_metrics,
        "finetuned_test_fixed_0p5_primary": final_fixed_metrics,
        "finetuned_test_validation_selected_thresholds_secondary": final_selected_metrics,
        "finetuned_minus_source_fixed_0p5": delta,
        "finetuned_fixed_0p5_patient_bootstrap": bootstrap_fixed,
        "finetuned_validation_threshold_patient_bootstrap": bootstrap_selected,
        "elapsed_seconds": time.time() - started,
    }
    write_json(output_dir / "test_metrics.json", summary)
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
