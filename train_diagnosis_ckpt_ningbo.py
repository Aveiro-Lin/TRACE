import argparse
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
from torch import nn
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset

from utils.downstream_trace import (
    TRACEDiagnosisClassifier,
    build_dataloaders,
    evaluate,
    save_checkpoint,
    train_one_epoch,
)

DEFAULT_NINGBO_ROOT = 'data/ningbo'
DEFAULT_NINGBO_LABEL_NAMES_JSON = 'data/ningbo_label_names_25.json'


def parse_ningbo_label_vector(value, expected_dim: int | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        labels = value.astype(np.float32, copy=False).reshape(-1)
    elif isinstance(value, (list, tuple)):
        labels = np.asarray(value, dtype=np.float32).reshape(-1)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            labels = np.zeros((0,), dtype=np.float32)
        else:
            cleaned = stripped.strip('[]').replace('\n', ' ')
            labels = np.fromstring(cleaned, sep=' ', dtype=np.float32)
            if labels.size == 0:
                labels = np.asarray(ast.literal_eval(stripped), dtype=np.float32).reshape(-1)
    elif pd.isna(value):
        labels = np.zeros((0,), dtype=np.float32)
    else:
        labels = np.asarray(value, dtype=np.float32).reshape(-1)

    if expected_dim is not None and labels.size != expected_dim:
        raise ValueError(f'Expected Ningbo label dim {expected_dim}, got {labels.size}.')
    return (labels > 0.0).astype(np.float32, copy=False)


def infer_ningbo_label_dim(index_csv: str | Path) -> int:
    df = pd.read_csv(index_csv, usecols=['LABEL'], nrows=1)
    if df.empty:
        raise ValueError(f'Cannot infer label dim from empty index: {index_csv}')
    return int(parse_ningbo_label_vector(df['LABEL'].iloc[0]).size)


def load_ningbo_label_names(label_names_json: str | Path | None, num_labels: int) -> list[str]:
    if label_names_json:
        path = Path(label_names_json)
        with path.open('r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, Mapping):
            payload = payload.get('diagnostic') or payload.get('labels') or payload.get('class_names') or payload.get('classes')
        label_names = [str(name) for name in payload]
        if len(label_names) != num_labels:
            raise ValueError(f'{path} has {len(label_names)} labels, expected {num_labels}.')
        return label_names
    return [f'Ningbo_{idx:02d}' for idx in range(num_labels)]


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
    return np.pad(signal, ((0, 0), (pad_left, pad_right)), mode='constant')


def resample_if_needed(signal: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return signal.astype(np.float32, copy=False)
    factor = gcd(int(source_rate), int(target_rate))
    up = int(target_rate) // factor
    down = int(source_rate) // factor
    return scipy.signal.resample_poly(signal, up, down, axis=-1).astype(np.float32, copy=False)


def extract_ningbo_window(signal: np.ndarray, seq_length: int, latent_ratio: float) -> np.ndarray:
    signal_length = int(signal.shape[-1])
    total_length = int(seq_length * (1.0 + latent_ratio))
    if signal_length >= total_length:
        start = (signal_length - total_length) // 2
    else:
        start = max(0, (signal_length - seq_length) // 2)
    segment = signal[:, start : start + seq_length]
    return center_crop_or_pad(segment, seq_length).astype(np.float32, copy=False)


def load_ningbo_signal(path: str | Path, channels: int) -> np.ndarray:
    with Path(path).open('rb') as f:
        signal = pickle.load(f)
    signal = np.asarray(signal, dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f'Expected 2D ECG array, got shape={signal.shape} from {path}')
    if signal.shape[0] != channels and signal.shape[1] == channels:
        signal = signal.T
    if signal.shape[0] < channels:
        raise ValueError(f'Expected at least {channels} leads, got shape={signal.shape} from {path}')
    return signal[:channels]


class NingboDownstreamDataset(Dataset):
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
        required_columns = {'FILE_NAME', 'LABEL'}
        missing_columns = sorted(required_columns.difference(df.columns))
        if missing_columns:
            raise ValueError(f'{self.index_csv} is missing required columns: {missing_columns}')
        if max_samples is not None:
            df = df.iloc[:max_samples].copy()
        self.df = df.reset_index(drop=True)

        self.records = [self._resolve_path(file_name) for file_name in self.df['FILE_NAME'].astype(str)]
        missing = [path for path in self.records if not path.exists()]
        if missing:
            raise FileNotFoundError(f'Missing Ningbo waveform file, first missing: {missing[0]}')

        self.source_rates = self._load_source_rates(self.df)
        label_dim = len(self.label_names)
        self.targets = np.stack(
            [parse_ningbo_label_vector(value, expected_dim=label_dim) for value in self.df['LABEL'].tolist()],
            axis=0,
        ).astype(np.float32, copy=False)

    def _resolve_path(self, file_name: str) -> Path:
        exact_path = self.data_dir / file_name
        if exact_path.exists():
            return exact_path
        basename_path = self.data_dir / Path(file_name).name
        if basename_path.exists():
            return basename_path
        return exact_path

    def _load_source_rates(self, df: pd.DataFrame) -> np.ndarray:
        if 'SAMPLE_RATE' not in df.columns:
            return np.full((len(df),), self.default_source_rate, dtype=np.int32)
        rates = []
        for value in df['SAMPLE_RATE'].tolist():
            rates.append(self.default_source_rate if pd.isna(value) else int(float(value)))
        return np.asarray(rates, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def label_counts(self) -> dict[str, int]:
        counts = self.targets.sum(axis=0).astype(int).tolist()
        return {label: count for label, count in zip(self.label_names, counts)}

    def summary(self) -> dict[str, object]:
        return {
            'index_csv': str(self.index_csv),
            'num_samples': len(self),
            'num_diagnostic_labels': len(self.label_names),
            'num_positive_labels': int(self.targets.sum()),
            'label_counts': self.label_counts,
        }

    def __getitem__(self, index: int):
        signal = load_ningbo_signal(self.records[index], channels=self.channels)
        signal = resample_if_needed(signal, int(self.source_rates[index]), self.target_rate)
        signal = extract_ningbo_window(signal, seq_length=self.seq_length, latent_ratio=self.latent_ratio)
        if self.normalize:
            signal = normalize_per_lead(signal)
        target = self.targets[index]
        return torch.from_numpy(signal.astype(np.float32, copy=False)), torch.from_numpy(target)


def parse_args():
    p = argparse.ArgumentParser('Train a Ningbo downstream diagnosis checkpoint with the diagnosis refinement branch')
    p.add_argument('--dataset', choices=['ningbo'], default='ningbo')
    p.add_argument(
        '--tokenizer-checkpoint',
        type=str,
        default='ckpt/ningbo/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_Ningbo_st2_best.pth',
    )
    p.add_argument('--base-downstream-checkpoint', type=str, default='')
    p.add_argument('--output-dir', type=str, default='ckpt/ningbo/downstream_refined')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--pooling', choices=['mean', 'flatten'], default='flatten')
    p.add_argument('--use-quantized-latents', action='store_true')
    p.add_argument('--freeze-backbone', action='store_true')
    p.add_argument('--train-base-branch', action='store_true')
    p.add_argument('--train-backbone-with-booster', action='store_true')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--target-metric', type=str, default='macro_auroc')
    p.add_argument('--disable-amp', action='store_true')
    p.add_argument('--seq-length', type=int, default=2250)
    p.add_argument('--patch-size', type=int, default=75)
    p.add_argument('--latent-ratio', type=float, default=0.1)
    p.add_argument('--channels', type=int, default=12)
    p.add_argument('--codebook-size', type=int, default=256)
    p.add_argument('--residual-levels', type=int, default=2)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--pin-memory', action='store_true')
    p.add_argument('--ningbo-root', type=str, default=DEFAULT_NINGBO_ROOT)
    p.add_argument('--train-index', type=str, default='train_index.csv')
    p.add_argument('--valid-index', type=str, default='val_index.csv')
    p.add_argument('--test-index', type=str, default='test_index.csv')
    p.add_argument('--label-names-json', type=str, default=DEFAULT_NINGBO_LABEL_NAMES_JSON)
    p.add_argument('--source-rate', type=int, default=500)
    p.add_argument('--target-rate', type=int, default=250)
    p.add_argument('--normalize', action='store_true', default=True)
    p.add_argument('--no-normalize', action='store_false', dest='normalize')
    p.add_argument('--max-train-samples', type=int, default=None)
    p.add_argument('--max-valid-samples', type=int, default=None)
    p.add_argument('--max-test-samples', type=int, default=None)
    return p.parse_args()


def apply_dataset_defaults(args):
    return args


def validate_dataset_routing(args):
    name = Path(args.tokenizer_checkpoint).name.lower()
    if 'ningbo' not in name:
        raise ValueError(f'Ningbo downstream ckpt should use a Ningbo tokenizer checkpoint, got: {args.tokenizer_checkpoint}')
    return args


def build_datasets(args):
    ningbo_root = Path(args.ningbo_root)
    train_index = ningbo_root / args.train_index
    valid_index = ningbo_root / args.valid_index
    test_index = ningbo_root / args.test_index

    label_dim = infer_ningbo_label_dim(train_index)
    class_names = load_ningbo_label_names(args.label_names_json, label_dim)
    common_kwargs = {
        'data_dir': ningbo_root / 'data',
        'label_names': class_names,
        'seq_length': args.seq_length,
        'latent_ratio': args.latent_ratio,
        'source_rate': args.source_rate,
        'target_rate': args.target_rate,
        'channels': args.channels,
        'normalize': args.normalize,
    }
    datasets = {
        'train': NingboDownstreamDataset(train_index, max_samples=args.max_train_samples, **common_kwargs),
        'valid': NingboDownstreamDataset(valid_index, max_samples=args.max_valid_samples, **common_kwargs),
        'test': NingboDownstreamDataset(test_index, max_samples=args.max_test_samples, **common_kwargs),
    }
    dataset_summary = {split: dataset.summary() for split, dataset in datasets.items()}
    return datasets, class_names, dataset_summary


def main():
    args = validate_dataset_routing(apply_dataset_defaults(parse_args()))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                'dataset': args.dataset,
                'tokenizer_checkpoint': args.tokenizer_checkpoint,
                'base_downstream_checkpoint': args.base_downstream_checkpoint,
                'output_dir': str(output_dir),
                'ningbo_root': args.ningbo_root,
                'train_base_branch': args.train_base_branch,
                'train_backbone_with_booster': args.train_backbone_with_booster,
                'residual_strategy': 'base_logits + residual_scale * delta_logits',
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    datasets, class_names, dataset_summary = build_datasets(args)
    print(json.dumps({'class_names': class_names, 'dataset_summary': dataset_summary}, ensure_ascii=False), flush=True)
    loaders = build_dataloaders(datasets=datasets, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)
    device = torch.device(args.device)
    use_amp = device.type == 'cuda' and not args.disable_amp
    model = TRACEDiagnosisClassifier(
        tokenizer_checkpoint=args.tokenizer_checkpoint,
        num_classes=len(class_names),
        seq_length=args.seq_length,
        patch_size=args.patch_size,
        latent_ratio=args.latent_ratio,
        channels=args.channels,
        codebook_size=args.codebook_size,
        residual_levels=args.residual_levels,
        pooling=args.pooling,
        dropout=args.dropout,
        quantize=args.use_quantized_latents,
        freeze_backbone=args.freeze_backbone or (bool(args.base_downstream_checkpoint) and not args.train_backbone_with_booster),
        base_downstream_checkpoint=args.base_downstream_checkpoint,
        freeze_base_branch=(bool(args.base_downstream_checkpoint) and not args.train_base_branch),
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    history_path = output_dir / 'history.jsonl'
    best_value = -float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, loaders['train'], optimizer, criterion, device, scaler, use_amp)
        valid_result = evaluate(model, loaders['valid'], criterion, device, args.threshold, use_amp)
        valid_metrics = valid_result['metrics']
        metric_value = float(valid_metrics.get(args.target_metric, -float('inf')))
        debug_metrics = model.booster_debug_stats()
        log_row = {
            'dataset': args.dataset,
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_result['loss'],
            **valid_metrics,
            **debug_metrics,
        }
        with history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + '\n')
        print(json.dumps(log_row, ensure_ascii=False), flush=True)
        metrics_payload = {'loss': valid_result['loss'], **valid_metrics, **debug_metrics}
        save_checkpoint(output_dir / 'last.pt', model, optimizer, epoch, metrics_payload, class_names, vars(args))
        if metric_value >= best_value:
            best_value = metric_value
            save_checkpoint(output_dir / 'best.pt', model, optimizer, epoch, metrics_payload, class_names, vars(args))
    summary = {
        'dataset': args.dataset,
        'output_dir': str(output_dir),
        'best_checkpoint': str(output_dir / 'best.pt'),
        'last_checkpoint': str(output_dir / 'last.pt'),
        'target_metric': args.target_metric,
        'epochs': args.epochs,
        'base_downstream_checkpoint': args.base_downstream_checkpoint,
        'tokenizer_checkpoint': args.tokenizer_checkpoint,
        'class_names': class_names,
        'dataset_summary': dataset_summary,
    }
    with (output_dir / 'ckpt_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
