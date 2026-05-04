
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from utils.report_controls_Ningbo import (
    DEFAULT_NINGBO_LABEL_NAMES_JSON,
    DEFAULT_NINGBO_ROOT,
    NingboReportControlDataset,
    ReportControlClassifier,
    build_control_loaders,
    build_ningbo_control_vocab,
    evaluate_control_epoch,
    infer_ningbo_label_dim,
    load_ningbo_label_names,
    save_control_checkpoint,
    save_control_vocab,
    train_control_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train the Ningbo report-control classifier")
    parser.add_argument(
        "--tokenizer-checkpoint",
        type=str,
        default="ckpt/ningbo/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_Ningbo_st2.pth",
    )
    parser.add_argument("--output-dir", type=str, default="ckpt/ningbo/report_control")
    parser.add_argument("--vocab-path", type=str, default="ckpt/ningbo/report_control/report_control_vocab_Ningbo.json")
    parser.add_argument("--ningbo-root", type=str, default=DEFAULT_NINGBO_ROOT)
    parser.add_argument("--train-index", type=str, default="train_index.csv")
    parser.add_argument("--valid-index", type=str, default="val_index.csv")
    parser.add_argument("--test-index", type=str, default="test_index.csv")
    parser.add_argument(
        "--label-names-json",
        type=str,
        default=DEFAULT_NINGBO_LABEL_NAMES_JSON,
        help="JSON list/dict for the 25 Ningbo diagnosis label names.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pooling", choices=["mean", "flatten"], default="flatten")
    parser.add_argument("--use-quantized-latents", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seq-length", type=int, default=2250)
    parser.add_argument("--patch-size", type=int, default=75)
    parser.add_argument("--latent-ratio", type=float, default=0.1)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--residual-levels", type=int, default=2)
    parser.add_argument("--source-rate", type=int, default=500)
    parser.add_argument("--target-rate", type=int, default=250)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--target-metric", type=str, default="mean_macro_f1")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def build_datasets(args: argparse.Namespace):
    ningbo_root = Path(args.ningbo_root)
    data_dir = ningbo_root / "data"
    train_index = resolve_under_root(ningbo_root, args.train_index)
    valid_index = resolve_under_root(ningbo_root, args.valid_index)
    test_index = resolve_under_root(ningbo_root, args.test_index)

    label_dim = infer_ningbo_label_dim(train_index)
    label_names = load_ningbo_label_names(args.label_names_json, label_dim)
    vocab = build_ningbo_control_vocab(label_names)
    save_control_vocab(args.vocab_path, vocab)

    common_kwargs = {
        "data_dir": data_dir,
        "label_names": label_names,
        "seq_length": args.seq_length,
        "latent_ratio": args.latent_ratio,
        "source_rate": args.source_rate,
        "target_rate": args.target_rate,
        "channels": args.channels,
        "normalize": args.normalize,
    }
    datasets = {
        "train": NingboReportControlDataset(
            train_index,
            max_samples=args.max_train_samples,
            **common_kwargs,
        ),
        "valid": NingboReportControlDataset(
            valid_index,
            max_samples=args.max_valid_samples,
            **common_kwargs,
        ),
        "test": NingboReportControlDataset(
            test_index,
            max_samples=args.max_test_samples,
            **common_kwargs,
        ),
    }
    dataset_summary = {split: dataset.summary() for split, dataset in datasets.items()}
    return datasets, vocab, dataset_summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets, vocab, dataset_summary = build_datasets(args)
    loaders = build_control_loaders(
        datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    setup_payload = {
        "tokenizer_checkpoint": args.tokenizer_checkpoint,
        "output_dir": str(output_dir),
        "vocab_path": args.vocab_path,
        "ningbo_root": args.ningbo_root,
        "control_column_map": {"diagnostic": "LABEL", "form": None, "rhythm": None},
        "vocab_sizes": {group: len(vocab.get(group, [])) for group in vocab.keys()},
        "dataset_summary": dataset_summary,
    }
    print(json.dumps(setup_payload, ensure_ascii=False), flush=True)

    if args.epochs <= 0:
        print(json.dumps({"status": "dataset_built_only", "epochs": args.epochs}, ensure_ascii=False), flush=True)
        return

    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.disable_amp
    model = ReportControlClassifier(
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        vocab=vocab,
        seq_length=args.seq_length,
        patch_size=args.patch_size,
        latent_ratio=args.latent_ratio,
        channels=args.channels,
        codebook_size=args.codebook_size,
        residual_levels=args.residual_levels,
        pooling=args.pooling,
        dropout=args.dropout,
        quantize=args.use_quantized_latents,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=use_amp)
    history_path = output_dir / "history.jsonl"

    best_value = -float("inf")
    for epoch in range(args.epochs):
        train_loss = train_control_epoch(model, loaders["train"], optimizer, criterion, device, scaler, use_amp)
        valid_result = evaluate_control_epoch(model, loaders["valid"], criterion, device, args.threshold, use_amp)
        valid_metrics = valid_result["metrics"]
        metric_value = float(valid_metrics.get(args.target_metric, -float("inf")))
        log_row = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_result["loss"], **valid_metrics}
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        print(json.dumps(log_row, ensure_ascii=False), flush=True)

        metrics_payload = {"loss": valid_result["loss"], **valid_metrics}
        args_dict = vars(args).copy()
        args_dict["control_column_map"] = setup_payload["control_column_map"]
        save_control_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            vocab,
            metrics_payload,
            args_dict,
            args.tokenizer_checkpoint,
        )
        if metric_value >= best_value:
            best_value = metric_value
            save_control_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                vocab,
                metrics_payload,
                args_dict,
                args.tokenizer_checkpoint,
            )

    best_checkpoint = torch.load(output_dir / "best.pt", map_location="cpu")
    model.load_state_dict(best_checkpoint["model_state_dict"])
    test_result = evaluate_control_epoch(model, loaders["test"], criterion, device, args.threshold, use_amp)
    payload = {
        "checkpoint": str(output_dir / "best.pt"),
        "vocab_path": args.vocab_path,
        "best_valid_metrics": best_checkpoint.get("metrics", {}),
        "test_loss": test_result["loss"],
        **test_result["metrics"],
    }
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
