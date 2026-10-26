
"""Train the TRACE ST2 tokenizer on the Ningbo ECG dataset.

This is a Ningbo-specific copy/adaptation of the tokenizer "beat_train" path.
It intentionally does not modify the existing PTB-XL or CPSC2018 scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from math import gcd
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.trace_tokenizer import Tokenizer


DEFAULT_NINGBO_ROOT = "data/ningbo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train ST2 ECG tokenizer on Ningbo")
    parser.add_argument("--ningbo-root", type=str, default=DEFAULT_NINGBO_ROOT)
    parser.add_argument("--train-index", type=str, default="train_index.csv")
    parser.add_argument("--valid-index", type=str, default="val_index.csv")
    parser.add_argument(
        "--output-path",
        type=str,
        default="ckpt/ningbo/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_Ningbo_st2.pth",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-length", type=int, default=2250)
    parser.add_argument("--patch-size", type=int, default=75)
    parser.add_argument("--latent-ratio", type=float, default=0.1)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--residual-levels", type=int, default=2)
    parser.add_argument("--source-rate", type=int, default=500)
    parser.add_argument("--target-rate", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_per_lead(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return (signal - mean) / (std + 1e-6)


def center_crop_or_pad(signal: np.ndarray, length: int) -> np.ndarray:
    current = signal.shape[-1]
    if current == length:
        return signal
    if current > length:
        start = (current - length) // 2
        return signal[:, start : start + length]

    pad_left = (length - current) // 2
    pad_right = length - current - pad_left
    return np.pad(signal, ((0, 0), (pad_left, pad_right)), mode="constant")


def resample_if_needed(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return signal
    factor = gcd(source_rate, target_rate)
    up = target_rate // factor
    down = source_rate // factor
    return scipy.signal.resample_poly(signal, up, down, axis=-1).astype(np.float32, copy=False)


class NingboSignalDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        data_dir: str | Path,
        total_length: int,
        source_rate: int,
        target_rate: int,
        channels: int,
        normalize: bool,
        max_samples: int | None = None,
    ) -> None:
        self.index_csv = Path(index_csv)
        self.data_dir = Path(data_dir)
        self.total_length = total_length
        self.source_rate = source_rate
        self.target_rate = target_rate
        self.channels = channels
        self.normalize = normalize

        df = pd.read_csv(self.index_csv)
        if max_samples is not None:
            df = df.iloc[:max_samples].copy()
        self.records = [self._resolve_path(file_name) for file_name in df["FILE_NAME"].astype(str)]
        missing = [path for path in self.records if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Ningbo waveform file, first missing: {missing[0]}")

    def _resolve_path(self, file_name: str) -> Path:
        exact_path = self.data_dir / file_name
        if exact_path.exists():
            return exact_path
        basename_path = self.data_dir / Path(file_name).name
        if basename_path.exists():
            return basename_path
        return exact_path

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor]:
        with self.records[index].open("rb") as f:
            signal = pickle.load(f)

        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim != 2:
            raise ValueError(f"Expected 2D ECG array, got shape={signal.shape} from {self.records[index]}")
        if signal.shape[0] != self.channels and signal.shape[1] == self.channels:
            signal = signal.T
        if signal.shape[0] < self.channels:
            raise ValueError(f"Expected at least {self.channels} leads, got shape={signal.shape}")
        signal = signal[: self.channels]

        signal = resample_if_needed(signal, self.source_rate, self.target_rate)
        signal = center_crop_or_pad(signal, self.total_length)
        if self.normalize:
            signal = normalize_per_lead(signal)
        return (torch.from_numpy(signal.astype(np.float32, copy=False)),)


def save_checkpoint(
    path: Path,
    model: Tokenizer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


@torch.no_grad()
def evaluate(model: Tokenizer, loader: DataLoader, seq_length: int, device: torch.device) -> dict[str, float]:
    model.eval()
    recon_loss_sum = 0.0
    pred_loss_sum = 0.0
    sample_count = 0

    for (signals,) in loader:
        signals = signals.to(device, non_blocking=True)
        recon_loss, _, pred_sequence = model(signals[:, :, :seq_length], predict=True)
        pred_target = signals[:, :, seq_length : seq_length + pred_sequence.shape[2]]
        pred_loss = F.mse_loss(pred_sequence, pred_target)
        batch_size = signals.size(0)
        recon_loss_sum += recon_loss.item() * batch_size
        pred_loss_sum += pred_loss.item() * batch_size
        sample_count += batch_size

    if sample_count == 0:
        return {"valid_recon_loss": 0.0, "valid_pred_loss": 0.0, "valid_total_loss": 0.0}
    valid_recon = recon_loss_sum / sample_count
    valid_pred = pred_loss_sum / sample_count
    return {
        "valid_recon_loss": valid_recon,
        "valid_pred_loss": valid_pred,
        "valid_total_loss": valid_recon + valid_pred,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)", flush=True)
    if torch.cuda.is_available():
        print(f"Current device index: {torch.cuda.current_device()}", flush=True)
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}", flush=True)

    device = torch.device(args.device)
    total_length = int(args.seq_length * (1 + args.latent_ratio))
    ningbo_root = Path(args.ningbo_root)
    data_dir = ningbo_root / "data"

    train_dataset = NingboSignalDataset(
        index_csv=ningbo_root / args.train_index,
        data_dir=data_dir,
        total_length=total_length,
        source_rate=args.source_rate,
        target_rate=args.target_rate,
        channels=args.channels,
        normalize=args.normalize,
        max_samples=args.max_train_samples,
    )
    valid_dataset = NingboSignalDataset(
        index_csv=ningbo_root / args.valid_index,
        data_dir=data_dir,
        total_length=total_length,
        source_rate=args.source_rate,
        target_rate=args.target_rate,
        channels=args.channels,
        normalize=args.normalize,
        max_samples=args.max_valid_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )

    tokenizer = Tokenizer(
        seq_length=args.seq_length,
        patch_size=args.patch_size,
        latent_ratio=args.latent_ratio,
        channels=args.channels,
        codebook_size=args.codebook_size,
        vq_kwargs={"residual_levels": args.residual_levels},
    ).to(device)
    optimizer = torch.optim.Adam(tokenizer.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    output_path = Path(args.output_path)
    last_path = output_path.with_name(f"{output_path.stem}_last{output_path.suffix}")
    best_path = output_path.with_name(f"{output_path.stem}_best{output_path.suffix}")
    history_path = output_path.with_name(f"{output_path.stem}_history.jsonl")
    run_name = output_path.stem
    writer = SummaryWriter(log_dir=str(Path("runs") / run_name))

    start_epoch = 0
    best_valid = float("inf")
    if output_path.exists():
        checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
        tokenizer.load_state_dict(checkpoint["model_state_dict"], strict=False)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_valid = float(checkpoint.get("metrics", {}).get("valid_total_loss", best_valid))
        print(f"Resumed Ningbo tokenizer training from epoch {start_epoch}", flush=True)

    print(
        json.dumps(
            {
                "ningbo_root": str(ningbo_root),
                "num_train": len(train_dataset),
                "num_valid": len(valid_dataset),
                "source_rate": args.source_rate,
                "target_rate": args.target_rate,
                "total_length": total_length,
                "output_path": str(output_path),
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for epoch in range(start_epoch, args.epochs):
        tokenizer.train()
        train_recon_sum = 0.0
        train_pred_sum = 0.0
        sample_count = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="batch", dynamic_ncols=True)
        for (signals,) in progress:
            signals = signals.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            recon_loss, _, pred_sequence = tokenizer(signals[:, :, : args.seq_length], predict=True)
            pred_target = signals[:, :, args.seq_length : args.seq_length + pred_sequence.shape[2]]
            pred_loss = F.mse_loss(pred_sequence, pred_target)
            loss = recon_loss + pred_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokenizer.parameters(), max_norm=1.0)
            optimizer.step()

            batch_size = signals.size(0)
            train_recon_sum += recon_loss.item() * batch_size
            train_pred_sum += pred_loss.item() * batch_size
            sample_count += batch_size
            progress.set_postfix(total_loss=f"{loss.item():.4f}")

        scheduler.step()

        train_recon = train_recon_sum / max(sample_count, 1)
        train_pred = train_pred_sum / max(sample_count, 1)
        train_total = train_recon + train_pred
        valid_metrics = evaluate(tokenizer, valid_loader, seq_length=args.seq_length, device=device)

        log_row = {
            "epoch": epoch,
            "train_recon_loss": train_recon,
            "train_pred_loss": train_pred,
            "train_total_loss": train_total,
            **valid_metrics,
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + "\n")
        writer.add_scalars("Loss/Total", {"Train": train_total, "Valid": valid_metrics["valid_total_loss"]}, epoch + 1)
        writer.add_scalars("Loss/Recon", {"Train": train_recon, "Valid": valid_metrics["valid_recon_loss"]}, epoch + 1)
        writer.add_scalars("Loss/Pred", {"Train": train_pred, "Valid": valid_metrics["valid_pred_loss"]}, epoch + 1)
        writer.flush()
        print(json.dumps(log_row, ensure_ascii=False), flush=True)

        save_checkpoint(last_path, tokenizer, optimizer, epoch, log_row, args)
        save_checkpoint(output_path, tokenizer, optimizer, epoch, log_row, args)
        if valid_metrics["valid_total_loss"] <= best_valid:
            best_valid = valid_metrics["valid_total_loss"]
            save_checkpoint(best_path, tokenizer, optimizer, epoch, log_row, args)

    writer.close()


if __name__ == "__main__":
    main()
