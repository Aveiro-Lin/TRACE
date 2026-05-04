from __future__ import annotations

import ast
import json
import pickle
from math import gcd
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import Dataset

from utils.report_controls import (
    CONTROL_GROUPS,
    ReportControlClassifier,
    build_control_loaders,
    evaluate_control_epoch,
    save_control_checkpoint,
    save_control_vocab,
    train_control_epoch,
)

DEFAULT_NINGBO_ROOT = "data/ningbo"
DEFAULT_NINGBO_LABEL_PREFIX = "Ningbo"
DEFAULT_NINGBO_LABEL_NAMES_JSON = "data/ningbo_label_names_25.json"


def parse_ningbo_label_vector(value, expected_dim: int | None = None) -> np.ndarray:
    """Parse the Ningbo index LABEL field into a float32 multi-hot vector."""
    if isinstance(value, np.ndarray):
        labels = value.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        labels = np.asarray(value, dtype=np.float32).reshape(-1)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            labels = np.zeros((0,), dtype=np.float32)
        else:
            cleaned = stripped.strip("[]").replace("\n", " ")
            labels = np.fromstring(cleaned, sep=" ", dtype=np.float32)
            if labels.size == 0:
                labels = np.asarray(ast.literal_eval(stripped), dtype=np.float32).reshape(-1)
    elif pd.isna(value):
        labels = np.zeros((0,), dtype=np.float32)
    else:
        labels = np.asarray(value, dtype=np.float32).reshape(-1)

    if expected_dim is not None and labels.size != expected_dim:
        raise ValueError(f"Expected Ningbo label dim {expected_dim}, got {labels.size}.")
    return (labels > 0.0).astype(np.float32, copy=False)


def infer_ningbo_label_dim(index_csv: str | Path) -> int:
    df = pd.read_csv(index_csv, usecols=["LABEL"], nrows=1)
    if df.empty:
        raise ValueError(f"Cannot infer label dim from empty index: {index_csv}")
    return int(parse_ningbo_label_vector(df["LABEL"].iloc[0]).size)


def load_ningbo_label_names(
    label_names_json: str | Path | None,
    num_labels: int,
    *,
    prefix: str = DEFAULT_NINGBO_LABEL_PREFIX,
) -> list[str]:
    if label_names_json:
        path = Path(label_names_json)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, Mapping):
            payload = (
                payload.get("diagnostic")
                or payload.get("labels")
                or payload.get("class_names")
                or payload.get("classes")
            )
        label_names = [str(name) for name in payload]
        if len(label_names) != num_labels:
            raise ValueError(f"{path} has {len(label_names)} labels, expected {num_labels}.")
        return label_names

    return [f"{prefix}_{idx:02d}" for idx in range(num_labels)]


def build_ningbo_control_vocab(label_names: Sequence[str]) -> dict[str, list[str]]:
    return {
        "diagnostic": [str(label) for label in label_names],
        "form": [],
        "rhythm": [],
    }


def normalize_per_lead(signal: np.ndarray) -> np.ndarray:
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return (signal - mean) / (std + 1e-6)


def center_crop_or_pad(signal: np.ndarray, length: int) -> np.ndarray:
    current = int(signal.shape[-1])
    if current == length:
        return signal
    if current > length:
        start = (current - length) // 2
        return signal[:, start : start + length]

    pad_left = (length - current) // 2
    pad_right = length - current - pad_left
    return np.pad(signal, ((0, 0), (pad_left, pad_right)), mode="constant")


def resample_if_needed(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return signal.astype(np.float32, copy=False)
    factor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // factor
    down = int(source_rate) // factor
    return scipy.signal.resample_poly(signal, up, down, axis=-1).astype(np.float32, copy=False)


def extract_ningbo_control_window(signal: np.ndarray, seq_length: int, latent_ratio: float) -> np.ndarray:
    signal_length = int(signal.shape[-1])
    total_length = int(seq_length * (1.0 + latent_ratio))
    if signal_length >= total_length:
        start = (signal_length - total_length) // 2
    else:
        start = max(0, (signal_length - seq_length) // 2)
    segment = signal[:, start : start + seq_length]
    return center_crop_or_pad(segment, seq_length).astype(np.float32, copy=False)


def load_ningbo_signal(path: str | Path, channels: int) -> np.ndarray:
    with Path(path).open("rb") as f:
        signal = pickle.load(f)

    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D ECG array, got shape={signal.shape} from {path}")
    if signal.shape[0] != channels and signal.shape[1] == channels:
        signal = signal.T
    if signal.shape[0] < channels:
        raise ValueError(f"Expected at least {channels} leads, got shape={signal.shape} from {path}")
    return signal[:channels]


class NingboReportControlDataset(Dataset):
    def __init__(
        self,
        index_csv: str | Path,
        data_dir: str | Path,
        *,
        label_names: Sequence[str],
        seq_length: int = 2250,
        latent_ratio: float = 0.1,
        source_rate: int = 500,
        target_rate: int = 250,
        channels: int = 12,
        normalize: bool = True,
        max_samples: int | None = None,
    ) -> None:
        self.index_csv = Path(index_csv)
        self.data_dir = Path(data_dir)
        self.label_names = [str(label) for label in label_names]
        self.seq_length = int(seq_length)
        self.latent_ratio = float(latent_ratio)
        self.default_source_rate = int(source_rate)
        self.target_rate = int(target_rate)
        self.channels = int(channels)
        self.normalize = bool(normalize)

        df = pd.read_csv(self.index_csv)
        required_columns = {"FILE_NAME", "LABEL"}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(f"{self.index_csv} is missing required columns: {missing_columns}")
        if max_samples is not None:
            df = df.iloc[:max_samples].copy()
        self.df = df.reset_index(drop=True)

        label_dim = len(self.label_names)
        self.records = [self._resolve_path(file_name) for file_name in self.df["FILE_NAME"].astype(str)]
        missing = [path for path in self.records if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing Ningbo waveform file, first missing: {missing[0]}")

        self.source_rates = self._load_source_rates(self.df)
        self.diagnostic_targets = np.stack(
            [parse_ningbo_label_vector(value, expected_dim=label_dim) for value in self.df["LABEL"].tolist()],
            axis=0,
        ).astype(np.float32, copy=False)
        self.form_targets = np.zeros((len(self.df), 0), dtype=np.float32)
        self.rhythm_targets = np.zeros((len(self.df), 0), dtype=np.float32)

    def _resolve_path(self, file_name: str) -> Path:
        exact_path = self.data_dir / file_name
        if exact_path.exists():
            return exact_path
        basename_path = self.data_dir / Path(file_name).name
        if basename_path.exists():
            return basename_path
        return exact_path

    def _load_source_rates(self, df: pd.DataFrame) -> np.ndarray:
        if "SAMPLE_RATE" not in df.columns:
            return np.full((len(df),), self.default_source_rate, dtype=np.int32)
        rates = []
        for value in df["SAMPLE_RATE"].tolist():
            if pd.isna(value):
                rates.append(self.default_source_rate)
            else:
                rates.append(int(float(value)))
        return np.asarray(rates, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def label_counts(self) -> dict[str, int]:
        counts = self.diagnostic_targets.sum(axis=0).astype(int).tolist()
        return {label: count for label, count in zip(self.label_names, counts)}

    def summary(self) -> dict[str, object]:
        return {
            "index_csv": str(self.index_csv),
            "num_samples": len(self),
            "num_diagnostic_labels": len(self.label_names),
            "num_positive_labels": int(self.diagnostic_targets.sum()),
            "label_counts": self.label_counts,
        }

    def __getitem__(self, index: int):
        signal = load_ningbo_signal(self.records[index], channels=self.channels)
        signal = resample_if_needed(signal, int(self.source_rates[index]), self.target_rate)
        signal = extract_ningbo_control_window(signal, seq_length=self.seq_length, latent_ratio=self.latent_ratio)
        if self.normalize:
            signal = normalize_per_lead(signal)

        return (
            torch.from_numpy(signal.astype(np.float32, copy=False)),
            torch.from_numpy(self.diagnostic_targets[index]),
            torch.from_numpy(self.form_targets[index]),
            torch.from_numpy(self.rhythm_targets[index]),
        )


__all__ = [
    "CONTROL_GROUPS",
    "DEFAULT_NINGBO_LABEL_NAMES_JSON",
    "DEFAULT_NINGBO_ROOT",
    "NingboReportControlDataset",
    "ReportControlClassifier",
    "build_control_loaders",
    "build_ningbo_control_vocab",
    "evaluate_control_epoch",
    "infer_ningbo_label_dim",
    "load_ningbo_label_names",
    "parse_ningbo_label_vector",
    "save_control_checkpoint",
    "save_control_vocab",
    "train_control_epoch",
]
