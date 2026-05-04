
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from utils import downstream_core as core_utils
from utils import downstream_trace as trace_utils
from utils.report_controls_Ningbo import (
    DEFAULT_NINGBO_LABEL_NAMES_JSON,
    DEFAULT_NINGBO_ROOT,
    NingboReportControlDataset,
    infer_ningbo_label_dim,
    load_ningbo_label_names,
)


DEFAULT_RATIOS = [0.05, 0.10, 0.25, 0.50, 1.00]


class NingboDiagnosisDataset(Dataset):
    def __init__(self, base: NingboReportControlDataset):
        self.base = base
        self.targets = base.diagnostic_targets
        self.indices = np.arange(len(base), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        signal, diagnostic, _form, _rhythm = self.base[index]
        return signal, diagnostic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train one label-efficiency downstream point.")
    parser.add_argument("--dataset", choices=["ptbxl", "cpsc2018", "ningbo"], required=True)
    parser.add_argument("--method", choices=["core", "trace"], required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default="label_efficiency_results/public")
    parser.add_argument("--tokenizer-checkpoint", type=str, default="")
    parser.add_argument("--base-downstream-checkpoint", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["mean", "flatten"], default="flatten")
    parser.add_argument("--use-quantized-latents", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--train-base-branch", action="store_true")
    parser.add_argument("--train-backbone-with-booster", action="store_true")
    parser.add_argument("--target-metric", type=str, default="macro_auroc")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--seq-length", type=int, default=2250)
    parser.add_argument("--patch-size", type=int, default=75)
    parser.add_argument("--latent-ratio", type=float, default=0.1)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--residual-levels", type=int, default=2)

    parser.add_argument("--metadata-csv", type=str, default="data/ptbxl_database.csv")
    parser.add_argument("--scp-statements-csv", type=str, default="data/scp_statements.csv")
    parser.add_argument("--signals-npy", type=str, default="data/records2250.npy")
    parser.add_argument("--valid-fold", type=int, default=9)
    parser.add_argument("--test-fold", type=int, default=10)
    parser.add_argument("--train-reference-csv", type=str, default="data/TRAIN_reference_cpsc2018.csv")
    parser.add_argument("--valid-reference-csv", type=str, default="data/VAL_reference_cpsc2018.csv")
    parser.add_argument("--test-reference-csv", type=str, default="data/TEST_reference_cpsc2018.csv")
    parser.add_argument("--train-signals-npy", type=str, default="data/records2250_cpsc2018_train.npy")
    parser.add_argument("--valid-signals-npy", type=str, default="data/records2250_cpsc2018_val.npy")
    parser.add_argument("--test-signals-npy", type=str, default="data/records2250_cpsc2018_test.npy")

    parser.add_argument("--ningbo-root", type=str, default=DEFAULT_NINGBO_ROOT)
    parser.add_argument("--ningbo-train-index", type=str, default="train_index.csv")
    parser.add_argument("--ningbo-valid-index", type=str, default="val_index.csv")
    parser.add_argument("--ningbo-test-index", type=str, default="test_index.csv")
    parser.add_argument("--ningbo-label-names-json", type=str, default=DEFAULT_NINGBO_LABEL_NAMES_JSON)
    parser.add_argument("--ningbo-source-rate", type=int, default=500)
    parser.add_argument("--ningbo-target-rate", type=int, default=250)
    parser.add_argument("--ningbo-no-normalize", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ratio_tag(ratio: float) -> str:
    pct = int(round(ratio * 100))
    return f"r{pct:03d}"


def apply_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if not args.tokenizer_checkpoint:
        if args.dataset == "ptbxl":
            args.tokenizer_checkpoint = "ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth"
        elif args.dataset == "cpsc2018":
            args.tokenizer_checkpoint = "ckpt/cpsc2018/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_cpsc2018_st2.pth"
        else:
            args.tokenizer_checkpoint = "ckpt/ningbo/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_Ningbo_st2_best.pth"

    if args.method == "trace" and not args.base_downstream_checkpoint:
        base_dir = Path(args.output_root) / args.dataset / "core" / ratio_tag(args.ratio) / f"seed_{args.seed}"
        args.base_downstream_checkpoint = str(base_dir / "best.pt")
    return args


def output_dir_for(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / args.dataset / args.method / ratio_tag(args.ratio) / f"seed_{args.seed}"


def build_datasets(args: argparse.Namespace):
    if args.dataset == "ptbxl":
        return trace_utils.build_ptbxl_datasets(
            metadata_csv=args.metadata_csv,
            scp_statements_csv=args.scp_statements_csv,
            signals_npy=args.signals_npy,
            seq_length=args.seq_length,
            latent_ratio=args.latent_ratio,
            valid_fold=args.valid_fold,
            test_fold=args.test_fold,
        )

    if args.dataset == "cpsc2018":
        return trace_utils.build_cpsc2018_datasets(
            train_reference_csv=args.train_reference_csv,
            valid_reference_csv=args.valid_reference_csv,
            test_reference_csv=args.test_reference_csv,
            train_signals_npy=args.train_signals_npy,
            valid_signals_npy=args.valid_signals_npy,
            test_signals_npy=args.test_signals_npy,
            seq_length=args.seq_length,
        )

    root = Path(args.ningbo_root)
    label_dim = infer_ningbo_label_dim(root / args.ningbo_train_index)
    class_names = load_ningbo_label_names(args.ningbo_label_names_json, label_dim)
    common_kwargs = {
        "data_dir": root / "data",
        "label_names": class_names,
        "seq_length": args.seq_length,
        "latent_ratio": args.latent_ratio,
        "source_rate": args.ningbo_source_rate,
        "target_rate": args.ningbo_target_rate,
        "channels": args.channels,
        "normalize": not args.ningbo_no_normalize,
    }
    datasets = {
        "train": NingboDiagnosisDataset(NingboReportControlDataset(root / args.ningbo_train_index, **common_kwargs)),
        "valid": NingboDiagnosisDataset(NingboReportControlDataset(root / args.ningbo_valid_index, **common_kwargs)),
        "test": NingboDiagnosisDataset(NingboReportControlDataset(root / args.ningbo_test_index, **common_kwargs)),
    }
    return datasets, class_names


def dataset_targets(dataset: Dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        targets = np.asarray(dataset.targets, dtype=np.float32)
        indices = getattr(dataset, "indices", None)
        if indices is not None:
            return targets[np.asarray(indices, dtype=np.int64)]
        return targets
    raise TypeError(f"Cannot extract targets from dataset type: {type(dataset)!r}")


def choose_label_subset(targets: np.ndarray, ratio: float, seed: int) -> np.ndarray:
    num_samples = int(targets.shape[0])
    if ratio >= 0.999:
        return np.arange(num_samples, dtype=np.int64)

    rng = np.random.default_rng(seed)
    sample_count = max(1, int(round(num_samples * ratio)))
    selected: set[int] = set()

    if sample_count >= targets.shape[1]:
        for class_idx in range(targets.shape[1]):
            positives = np.flatnonzero(targets[:, class_idx] > 0)
            if positives.size:
                selected.add(int(rng.choice(positives)))
            if len(selected) >= sample_count:
                break

    remaining = [idx for idx in rng.permutation(num_samples).tolist() if idx not in selected]
    selected.update(remaining[: max(0, sample_count - len(selected))])
    return np.asarray(sorted(selected), dtype=np.int64)


def apply_train_ratio(datasets: dict[str, Dataset], ratio: float, seed: int):
    train_targets = dataset_targets(datasets["train"])
    selected = choose_label_subset(train_targets, ratio, seed)
    subset_targets = train_targets[selected]
    datasets = dict(datasets)
    datasets["train"] = Subset(datasets["train"], selected.tolist())
    summary = {
        "train_total": int(train_targets.shape[0]),
        "train_selected": int(len(selected)),
        "ratio_requested": float(ratio),
        "ratio_actual": float(len(selected) / max(train_targets.shape[0], 1)),
        "positive_counts": subset_targets.sum(axis=0).astype(int).tolist(),
        "selected_indices": selected.tolist(),
    }
    return datasets, summary


def make_train_args(args: argparse.Namespace, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        base_downstream_checkpoint=args.base_downstream_checkpoint,
        output_dir=str(output_dir),
        downstream_checkpoint="",
        eval_only=False,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        pooling=args.pooling,
        use_quantized_latents=args.use_quantized_latents,
        freeze_backbone=args.freeze_backbone,
        train_base_branch=args.train_base_branch,
        train_backbone_with_booster=args.train_backbone_with_booster,
        device=args.device,
        threshold=args.threshold,
        target_metric=args.target_metric,
        disable_amp=args.disable_amp,
        seq_length=args.seq_length,
        patch_size=args.patch_size,
        latent_ratio=args.latent_ratio,
        channels=args.channels,
        codebook_size=args.codebook_size,
        residual_levels=args.residual_levels,
        pin_memory=args.pin_memory,
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = apply_defaults(parse_args())
    seed_everything(args.seed)

    output_dir = output_dir_for(args)
    metrics_path = output_dir / "test_metrics.json"
    if metrics_path.exists() and not args.force:
        print(json.dumps({"status": "skip_existing", "metrics_path": str(metrics_path)}, ensure_ascii=False))
        return

    if args.method == "trace" and not Path(args.base_downstream_checkpoint).exists():
        raise FileNotFoundError(f"TRACE needs the same-ratio base checkpoint: {args.base_downstream_checkpoint}")

    datasets, class_names = build_datasets(args)
    datasets, subset_summary = apply_train_ratio(datasets, ratio=args.ratio, seed=args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = subset_summary.pop("selected_indices", [])
    selected_indices_path = output_dir / "selected_indices.json"
    write_json(selected_indices_path, {"selected_indices": selected_indices})
    subset_summary["selected_indices_path"] = str(selected_indices_path)
    setup_payload = {
        "dataset": args.dataset,
        "method": args.method,
        "ratio": args.ratio,
        "seed": args.seed,
        "class_names": list(class_names),
        "tokenizer_checkpoint": args.tokenizer_checkpoint,
        "base_downstream_checkpoint": args.base_downstream_checkpoint,
        "subset_summary": subset_summary,
        "args": vars(args),
    }
    write_json(output_dir / "setup.json", setup_payload)
    print(json.dumps(setup_payload, ensure_ascii=False), flush=True)

    train_args = make_train_args(args, output_dir)
    if args.method == "core":
        payload = core_utils.run_downstream_experiment(
            train_args,
            datasets=datasets,
            class_names=class_names,
            task_name=f"{args.dataset}_label_efficiency_core",
        )
    else:
        payload = trace_utils.run_downstream_experiment(
            train_args,
            datasets=datasets,
            class_names=class_names,
            task_name=f"{args.dataset}_label_efficiency_trace",
        )

    payload.update(
        {
            "dataset": args.dataset,
            "method": args.method,
            "ratio": args.ratio,
            "seed": args.seed,
            "tokenizer_checkpoint": args.tokenizer_checkpoint,
            "base_downstream_checkpoint": args.base_downstream_checkpoint,
            "subset_summary": subset_summary,
        }
    )
    write_json(metrics_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
