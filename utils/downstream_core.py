from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.checkpoint_compat import remap_legacy_foundation_encoder_keys
from utils.trace_tokenizer import Tokenizer

PTBXL_SUPERCLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']
CPSC2018_CLASSES = ['Normal', 'AF', 'I-AVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _macro_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        target = y_true[:, i]
        pred = y_pred[:, i]
        tn = float(np.sum((target == 0) & (pred == 0)))
        fp = float(np.sum((target == 0) & (pred == 1)))
        denom = tn + fp
        if denom > 0:
            scores.append(tn / denom)
    return float(np.mean(scores)) if scores else float('nan')


def _macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    aucs = []
    for i in range(y_true.shape[1]):
        target = y_true[:, i]
        if np.unique(target).size < 2:
            continue
        aucs.append(roc_auc_score(target, y_prob[:, i]))
    return float(np.mean(aucs)) if aucs else float('nan')


class IndexedECGDataset(Dataset):
    def __init__(self, signals: np.ndarray, targets: np.ndarray, indices: np.ndarray | None = None):
        self.signals = signals
        self.targets = targets.astype(np.float32)
        self.indices = np.arange(len(targets)) if indices is None else np.asarray(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        real_index = int(self.indices[index])
        signal = np.asarray(self.signals[real_index], dtype=np.float32)
        target = np.asarray(self.targets[real_index], dtype=np.float32)
        return torch.from_numpy(signal), torch.from_numpy(target)


def slice_signal_window(signals: np.ndarray, seq_length: int, latent_ratio: float) -> np.ndarray:
    signal_length = int(signals.shape[-1])
    if signal_length < seq_length:
        raise ValueError(f'Signal length {signal_length} is shorter than seq_length={seq_length}.')

    total_length = int(seq_length * (1.0 + latent_ratio))
    if signal_length >= total_length:
        start = max(0, (signal_length - total_length) // 2)
    else:
        start = max(0, (signal_length - seq_length) // 2)

    end = start + seq_length
    if end > signal_length:
        end = signal_length
        start = end - seq_length

    return signals[:, :, start:end]


def labels_to_multihot(label_lists: Sequence[Sequence[str]], class_names: Sequence[str]) -> np.ndarray:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    targets = np.zeros((len(label_lists), len(class_names)), dtype=np.float32)
    for row_idx, labels in enumerate(label_lists):
        for label in labels:
            if label in class_to_idx:
                targets[row_idx, class_to_idx[label]] = 1.0
    return targets


def build_ptbxl_datasets(
    metadata_csv: str,
    scp_statements_csv: str,
    signals_npy: str,
    seq_length: int = 2250,
    latent_ratio: float = 0.1,
    valid_fold: int = 9,
    test_fold: int = 10,
) -> Tuple[Dict[str, Dataset], List[str]]:
    metadata = pd.read_csv(metadata_csv, index_col='ecg_id')
    metadata['scp_codes'] = metadata['scp_codes'].apply(ast.literal_eval)

    agg_df = pd.read_csv(scp_statements_csv, index_col=0)
    agg_df = agg_df[agg_df['diagnostic'] == 1.0]

    def aggregate_diagnostic(codes: Mapping[str, float]) -> List[str]:
        labels = []
        for key in codes.keys():
            if key in agg_df.index:
                labels.append(str(agg_df.loc[key, 'diagnostic_class']))
        return sorted(set(labels))

    metadata['diagnostic_superclass'] = metadata['scp_codes'].apply(aggregate_diagnostic)
    label_lists = metadata['diagnostic_superclass'].tolist()
    targets = labels_to_multihot(label_lists, PTBXL_SUPERCLASSES)
    keep_mask = targets.sum(axis=1) > 0

    signals = np.load(signals_npy, mmap_mode='r')
    signals = slice_signal_window(signals, seq_length=seq_length, latent_ratio=latent_ratio)

    if signals.shape[0] != len(metadata):
        raise ValueError('PTB-XL signals and metadata have different sample counts.')

    train_idx = np.where((metadata['strat_fold'] != valid_fold) & (metadata['strat_fold'] != test_fold) & keep_mask)[0]
    valid_idx = np.where((metadata['strat_fold'] == valid_fold) & keep_mask)[0]
    test_idx = np.where((metadata['strat_fold'] == test_fold) & keep_mask)[0]

    datasets = {
        'train': IndexedECGDataset(signals, targets, train_idx),
        'valid': IndexedECGDataset(signals, targets, valid_idx),
        'test': IndexedECGDataset(signals, targets, test_idx),
    }
    return datasets, list(PTBXL_SUPERCLASSES)


def _reference_to_targets(reference_csv: str) -> np.ndarray:
    df = pd.read_csv(reference_csv)
    targets = np.zeros((len(df), len(CPSC2018_CLASSES)), dtype=np.float32)
    label_cols = [col for col in df.columns if col.lower().endswith('label')]

    for row_idx, row in df.iterrows():
        for col in label_cols:
            value = row[col]
            if pd.isna(value) or str(value).strip() == '':
                continue
            label_index = int(value) - 1
            if 0 <= label_index < len(CPSC2018_CLASSES):
                targets[row_idx, label_index] = 1.0
    return targets


def build_cpsc2018_datasets(
    train_reference_csv: str,
    valid_reference_csv: str,
    test_reference_csv: str,
    train_signals_npy: str,
    valid_signals_npy: str,
    test_signals_npy: str,
    seq_length: int = 2250,
) -> Tuple[Dict[str, Dataset], List[str]]:
    train_signals = np.load(train_signals_npy, mmap_mode='r')[:, :, :seq_length]
    valid_signals = np.load(valid_signals_npy, mmap_mode='r')[:, :, :seq_length]
    test_signals = np.load(test_signals_npy, mmap_mode='r')[:, :, :seq_length]

    train_targets = _reference_to_targets(train_reference_csv)
    valid_targets = _reference_to_targets(valid_reference_csv)
    test_targets = _reference_to_targets(test_reference_csv)

    datasets = {
        'train': IndexedECGDataset(train_signals, train_targets),
        'valid': IndexedECGDataset(valid_signals, valid_targets),
        'test': IndexedECGDataset(test_signals, test_targets),
    }
    return datasets, list(CPSC2018_CLASSES)


def compute_multilabel_metrics(logits: np.ndarray, targets: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    probabilities = _sigmoid(logits)
    predictions = (probabilities >= threshold).astype(np.int64)
    targets = targets.astype(np.int64)

    macro_accuracy = float(np.mean((predictions == targets).mean(axis=0)))
    subset_accuracy = float(np.all(predictions == targets, axis=1).mean())

    metrics = {
        'macro_accuracy': macro_accuracy,
        'subset_accuracy': subset_accuracy,
        'macro_precision': float(precision_score(targets, predictions, average='macro', zero_division=0)),
        'macro_recall': float(recall_score(targets, predictions, average='macro', zero_division=0)),
        'macro_f1': float(f1_score(targets, predictions, average='macro', zero_division=0)),
        'macro_auroc': _macro_auroc(targets, probabilities),
        'macro_specificity': _macro_specificity(targets, predictions),
    }
    return metrics


class TRACEDiagnosisClassifier(nn.Module):
    def __init__(
        self,
        tokenizer_checkpoint: str,
        num_classes: int,
        seq_length: int = 2250,
        patch_size: int = 75,
        latent_ratio: float = 0.1,
        channels: int = 12,
        codebook_size: int = 256,
        residual_levels: int = 2,
        pooling: str = 'flatten',
        dropout: float = 0.1,
        quantize: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.quantize = quantize
        self.freeze_backbone = freeze_backbone

        self.backbone = Tokenizer(
            seq_length=seq_length,
            patch_size=patch_size,
            latent_ratio=latent_ratio,
            channels=channels,
            codebook_size=codebook_size,
            freeze_foundation_encoder=freeze_backbone,
            vq_kwargs={'residual_levels': residual_levels},
        )
        checkpoint = torch.load(tokenizer_checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        state_dict, remapped_keys = remap_legacy_foundation_encoder_keys(state_dict)
        if remapped_keys:
            print(f'Remapped {remapped_keys} legacy tokenizer checkpoint keys.')
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            print('Tokenizer load info:', msg)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if pooling == 'mean':
            feature_dim = self.backbone.dim
        elif pooling == 'flatten':
            feature_dim = self.backbone.dim * self.backbone.num_latent_tokens
        else:
            raise ValueError(f'Unsupported pooling mode: {pooling}')

        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feature_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.encode_features(x, pooling=self.pooling, quantize=self.quantize)
        features = self.norm(features)
        features = self.dropout(features)
        return self.head(features)


def build_dataloaders(
    datasets: Dict[str, Dataset],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Dict[str, DataLoader]:
    return {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        'valid': DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler,
    use_amp: bool,
) -> float:
    model.train()
    running_loss = 0.0
    sample_count = 0

    for signals, targets in tqdm(loader, desc='train', leave=False):
        signals = signals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(signals)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = signals.size(0)
        running_loss += loss.detach().item() * batch_size
        sample_count += batch_size

    return running_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float,
    use_amp: bool,
) -> Dict[str, object]:
    model.eval()
    running_loss = 0.0
    sample_count = 0
    logits_buffer = []
    targets_buffer = []

    for signals, targets in tqdm(loader, desc='eval', leave=False):
        signals = signals.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(signals)
            loss = criterion(logits, targets)

        batch_size = signals.size(0)
        running_loss += loss.detach().item() * batch_size
        sample_count += batch_size
        logits_buffer.append(logits.detach().cpu())
        targets_buffer.append(targets.detach().cpu())

    logits = torch.cat(logits_buffer, dim=0).numpy()
    targets = torch.cat(targets_buffer, dim=0).numpy()
    metrics = compute_multilabel_metrics(logits, targets, threshold=threshold)
    return {
        'loss': running_loss / max(sample_count, 1),
        'metrics': metrics,
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    class_names: Sequence[str],
    args_dict: Dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'class_names': list(class_names),
            'args': args_dict,
        },
        path,
    )


def load_downstream_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, object]:
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def add_common_args(parser, default_tokenizer_checkpoint: str, default_output_dir: str):
    parser.add_argument('--tokenizer-checkpoint', type=str, default=default_tokenizer_checkpoint)
    parser.add_argument('--output-dir', type=str, default=default_output_dir)
    parser.add_argument('--downstream-checkpoint', type=str, default='')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling', choices=['mean', 'flatten'], default='flatten')
    parser.add_argument('--use-quantized-latents', action='store_true')
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--target-metric', type=str, default='macro_auroc')
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--seq-length', type=int, default=2250)
    parser.add_argument('--patch-size', type=int, default=75)
    parser.add_argument('--latent-ratio', type=float, default=0.1)
    parser.add_argument('--channels', type=int, default=12)
    parser.add_argument('--codebook-size', type=int, default=256)
    parser.add_argument('--residual-levels', type=int, default=2)
    parser.add_argument('--pin-memory', action='store_true')
    return parser


def run_downstream_experiment(args, datasets: Dict[str, Dataset], class_names: Sequence[str], task_name: str) -> Dict[str, object]:
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        datasets=datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

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
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == 'cuda' and not args.disable_amp
    scaler = GradScaler(enabled=use_amp)

    checkpoint_path = Path(args.downstream_checkpoint) if args.downstream_checkpoint else output_dir / 'best.pt'

    if args.eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
        checkpoint = load_downstream_checkpoint(checkpoint_path, model)
        test_result = evaluate(model, loaders['test'], criterion, device, args.threshold, use_amp)
        payload = {
            'task': task_name,
            'checkpoint': str(checkpoint_path),
            'class_names': list(class_names),
            'loaded_metrics': checkpoint.get('metrics', {}),
            'test_loss': test_result['loss'],
            **test_result['metrics'],
        }
        save_json(output_dir / 'test_metrics.json', payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return payload

    best_value = -float('inf')
    history_path = output_dir / 'history.jsonl'

    if checkpoint_path.exists():
        checkpoint = load_downstream_checkpoint(checkpoint_path, model, optimizer)
        print(f'Resumed downstream checkpoint from {checkpoint_path}')
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        previous_metrics = checkpoint.get('metrics', {})
        if args.target_metric in previous_metrics:
            best_value = float(previous_metrics[args.target_metric])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, loaders['train'], optimizer, criterion, device, scaler, use_amp)
        valid_result = evaluate(model, loaders['valid'], criterion, device, args.threshold, use_amp)
        valid_metrics = valid_result['metrics']
        metric_value = float(valid_metrics.get(args.target_metric, -float('inf')))

        log_row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_result['loss'],
            **valid_metrics,
        }
        with history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + '\n')

        print(json.dumps(log_row, ensure_ascii=False))

        if metric_value >= best_value:
            best_value = metric_value
            save_checkpoint(
                output_dir / 'best.pt',
                model,
                optimizer,
                epoch,
                {'loss': valid_result['loss'], **valid_metrics},
                class_names,
                vars(args),
            )

        save_checkpoint(
            output_dir / 'last.pt',
            model,
            optimizer,
            epoch,
            {'loss': valid_result['loss'], **valid_metrics},
            class_names,
            vars(args),
        )

    best_checkpoint = load_downstream_checkpoint(output_dir / 'best.pt', model)
    test_result = evaluate(model, loaders['test'], criterion, device, args.threshold, use_amp)
    payload = {
        'task': task_name,
        'checkpoint': str(output_dir / 'best.pt'),
        'class_names': list(class_names),
        'best_valid_metrics': best_checkpoint.get('metrics', {}),
        'test_loss': test_result['loss'],
        **test_result['metrics'],
    }
    save_json(output_dir / 'test_metrics.json', payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload
