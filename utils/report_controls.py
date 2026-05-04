from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.trace_tokenizer import Tokenizer
from utils.downstream_trace import (
    CPSC2018_CLASSES,
    PTBXL_SUPERCLASSES,
    TRACEDiagnosisClassifier,
)

CONTROL_GROUPS = ('diagnostic', 'form', 'rhythm')
DEFAULT_CONTROL_COLUMN_MAP = {group: group for group in CONTROL_GROUPS}


def parse_control_dict(value) -> Dict[str, float]:
    if isinstance(value, Mapping):
        parsed = value
    elif value is None:
        parsed = {}
    elif isinstance(value, float) and np.isnan(value):
        parsed = {}
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            parsed = {}
        else:
            parsed = ast.literal_eval(stripped)
    else:
        parsed = value

    if not isinstance(parsed, Mapping):
        return {}
    return {str(key): float(parsed[key]) for key in parsed.keys()}


def extract_signal_window(signals: np.ndarray, seq_length: int, latent_ratio: float) -> np.ndarray:
    total_length = int(seq_length * (1.0 + latent_ratio))
    start = int((signals.shape[2] - total_length) / 2)
    return signals[:, :, start:start + seq_length]


def handle_ptbxl_sex(sex):
    if isinstance(sex, str):
        return sex
    if isinstance(sex, int):
        return 'male' if sex == 1 else 'female'
    return 'unknown'


def compute_ptbxl_label_aggregations(df: pd.DataFrame, scp_statements_csv: str) -> pd.DataFrame:
    symptoms = list(CONTROL_GROUPS)
    aggregation_df = pd.read_csv(scp_statements_csv, index_col=0)
    agg_df = {symptom: aggregation_df[aggregation_df[symptom] == 1.0] for symptom in symptoms}

    def aggregate(symptom: str):
        def _aggregate(y_dic: Mapping[str, float]):
            tmp = {}
            for key in y_dic.keys():
                if key in agg_df[symptom].index:
                    tmp.update([(key, y_dic[key])])
            return str(tmp)
        return _aggregate

    df = df.copy()
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))
    for symptom in symptoms:
        df[symptom] = df.scp_codes.apply(aggregate(symptom=symptom))
    return df


def _normalize_control_column_map(column_map: Mapping[str, str] | None = None) -> Dict[str, str]:
    resolved = dict(DEFAULT_CONTROL_COLUMN_MAP)
    if column_map is not None:
        resolved.update({str(group): str(column) for group, column in column_map.items()})
    return resolved


def add_ptbxl_diagnostic_superclass_column(df: pd.DataFrame, scp_statements_csv: str) -> pd.DataFrame:
    aggregation_df = pd.read_csv(scp_statements_csv, index_col=0)
    diagnostic_df = aggregation_df[aggregation_df['diagnostic'] == 1.0]

    def _to_superclass_dict(scp_codes: Mapping[str, float]) -> str:
        parsed = scp_codes if isinstance(scp_codes, Mapping) else parse_control_dict(scp_codes)
        result = {}
        for code, score in parsed.items():
            if code not in diagnostic_df.index:
                continue
            diagnostic_class = diagnostic_df.loc[code, 'diagnostic_class']
            if not isinstance(diagnostic_class, str) or not diagnostic_class or diagnostic_class == 'nan':
                continue
            current = result.get(diagnostic_class)
            result[diagnostic_class] = float(score) if current is None else max(current, float(score))
        return str(result)

    df = df.copy()
    df['diagnostic_superclass'] = df['scp_codes'].apply(_to_superclass_dict)
    return df


def sanitize_control_name(name: str) -> str:
    sanitized = ''.join(ch.lower() if ch.isalnum() else '_' for ch in str(name))
    sanitized = '_'.join(part for part in sanitized.split('_') if part)
    if not sanitized:
        sanitized = 'unknown'
    if sanitized[0].isdigit():
        sanitized = f'label_{sanitized}'
    return sanitized


def build_control_vocab_from_dataframe(
    df: pd.DataFrame,
    *,
    column_map: Mapping[str, str] | None = None,
) -> Dict[str, list[str]]:
    column_map = _normalize_control_column_map(column_map)
    vocab = {group: set() for group in CONTROL_GROUPS}
    for group in CONTROL_GROUPS:
        source_column = column_map[group]
        if source_column not in df.columns:
            continue
        for value in df[source_column].tolist():
            vocab[group].update(parse_control_dict(value).keys())
    return {group: sorted(values) for group, values in vocab.items()}


def _count_control_labels(values: Sequence[object]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        for label in parse_control_dict(value).keys():
            counts[label] = counts.get(label, 0) + 1
    return counts


def build_ptbxl_v15_2_vocab(
    train_df: pd.DataFrame,
    *,
    min_form_count: int = 300,
    min_rhythm_count: int = 100,
) -> Dict[str, list[str]]:
    diagnostic_counts = _count_control_labels(train_df['diagnostic_superclass'].tolist())
    form_counts = _count_control_labels(train_df['form'].tolist())
    rhythm_counts = _count_control_labels(train_df['rhythm'].tolist())

    def _sorted_labels(counts: Mapping[str, int], minimum: int) -> list[str]:
        selected = [(label, count) for label, count in counts.items() if count >= minimum]
        selected.sort(key=lambda item: (-item[1], item[0]))
        return [label for label, _ in selected]

    return {
        'diagnostic': _sorted_labels(diagnostic_counts, 1),
        'form': _sorted_labels(form_counts, min_form_count),
        'rhythm': _sorted_labels(rhythm_counts, min_rhythm_count),
    }


def build_ptbxl_v15_vocab(
    train_df: pd.DataFrame,
    *,
    min_form_count: int = 300,
    min_rhythm_count: int = 100,
) -> Dict[str, list[str]]:
    return build_ptbxl_v15_2_vocab(
        train_df,
        min_form_count=min_form_count,
        min_rhythm_count=min_rhythm_count,
    )


def save_control_vocab(path: str | Path, vocab: Mapping[str, Sequence[str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump({group: list(labels) for group, labels in vocab.items()}, f, ensure_ascii=False, indent=2)


def load_control_vocab(path: str | Path) -> Dict[str, list[str]]:
    with Path(path).open('r', encoding='utf-8') as f:
        data = json.load(f)
    return {group: list(data.get(group, [])) for group in CONTROL_GROUPS}


def build_control_token_maps(vocab: Mapping[str, Sequence[str]]) -> Dict[str, object]:
    token_maps = {}
    special_tokens = []
    used_tokens = set()
    for group in CONTROL_GROUPS:
        start_token = f'<|{group}_start|>'
        end_token = f'<|{group}_end|>'
        none_token = f'<|{group}_none|>'
        label_to_token = {}
        for label in vocab.get(group, []):
            token = f'<|{group}_{sanitize_control_name(label)}|>'
            if token in used_tokens:
                raise ValueError(f'Duplicate control token generated for label "{label}": {token}')
            used_tokens.add(token)
            label_to_token[label] = token
        token_maps[group] = {
            'start': start_token,
            'end': end_token,
            'none': none_token,
            'label_to_token': label_to_token,
        }
        special_tokens.extend([start_token, end_token, none_token, *label_to_token.values()])
    return {'groups': token_maps, 'special_tokens': special_tokens}


def build_control_targets(
    df: pd.DataFrame,
    vocab: Mapping[str, Sequence[str]],
    *,
    column_map: Mapping[str, str] | None = None,
) -> Dict[str, np.ndarray]:
    column_map = _normalize_control_column_map(column_map)
    targets = {}
    for group in CONTROL_GROUPS:
        labels = list(vocab.get(group, []))
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        target = np.zeros((len(df), len(labels)), dtype=np.float32)
        source_column = column_map[group]
        if source_column not in df.columns:
            targets[group] = target
            continue
        for row_idx, value in enumerate(df[source_column].tolist()):
            for label in parse_control_dict(value).keys():
                if label in label_to_index:
                    target[row_idx, label_to_index[label]] = 1.0
        targets[group] = target
    return targets


def render_control_prompt(
    label_dicts: Mapping[str, Mapping[str, float]],
    vocab: Mapping[str, Sequence[str]],
    token_maps: Mapping[str, object] | None = None,
) -> str:
    if token_maps is None:
        token_maps = build_control_token_maps(vocab)['groups']

    sections = []
    for group in CONTROL_GROUPS:
        group_tokens = token_maps[group]
        sections.append(group_tokens['start'])
        selected_labels = [label for label in vocab.get(group, []) if label in label_dicts.get(group, {})]
        if selected_labels:
            sections.extend(group_tokens['label_to_token'][label] for label in selected_labels)
        else:
            sections.append(group_tokens['none'])
        sections.append(group_tokens['end'])
    return ' '.join(sections)


def render_control_prompt_from_row(
    row,
    vocab: Mapping[str, Sequence[str]],
    token_maps: Mapping[str, object] | None = None,
    column_map: Mapping[str, str] | None = None,
) -> str:
    column_map = _normalize_control_column_map(column_map)
    label_dicts = {
        group: parse_control_dict(row[column_map[group]]) if column_map[group] in row.index else {}
        for group in CONTROL_GROUPS
    }
    return render_control_prompt(label_dicts, vocab=vocab, token_maps=token_maps)


def render_control_prompt_from_binary(
    binary_predictions: Mapping[str, np.ndarray],
    vocab: Mapping[str, Sequence[str]],
    token_maps: Mapping[str, object] | None = None,
) -> str:
    label_dicts = {}
    for group in CONTROL_GROUPS:
        labels = list(vocab.get(group, []))
        selected = [labels[idx] for idx, flag in enumerate(binary_predictions[group].tolist()) if flag]
        label_dicts[group] = {label: 100.0 for label in selected}
    return render_control_prompt(label_dicts, vocab=vocab, token_maps=token_maps)


class ReportControlDataset(Dataset):
    def __init__(
        self,
        signals: np.ndarray,
        diagnostic_targets: np.ndarray,
        form_targets: np.ndarray,
        rhythm_targets: np.ndarray,
    ):
        self.signals = signals
        self.diagnostic_targets = diagnostic_targets.astype(np.float32)
        self.form_targets = form_targets.astype(np.float32)
        self.rhythm_targets = rhythm_targets.astype(np.float32)

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, index: int):
        return (
            torch.as_tensor(np.asarray(self.signals[index], dtype=np.float32)),
            torch.as_tensor(np.asarray(self.diagnostic_targets[index], dtype=np.float32)),
            torch.as_tensor(np.asarray(self.form_targets[index], dtype=np.float32)),
            torch.as_tensor(np.asarray(self.rhythm_targets[index], dtype=np.float32)),
        )


class ReportControlClassifier(nn.Module):
    def __init__(
        self,
        *,
        tokenizer_checkpoint: str,
        vocab: Mapping[str, Sequence[str]],
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
        self.vocab = {group: list(vocab.get(group, [])) for group in CONTROL_GROUPS}

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
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            print('Tokenizer load info:', msg)

        if pooling == 'mean':
            feature_dim = self.backbone.dim
        elif pooling == 'flatten':
            feature_dim = self.backbone.dim * self.backbone.num_latent_tokens
        else:
            raise ValueError(f'Unsupported pooling mode: {pooling}')

        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.diagnostic_head = nn.Linear(feature_dim, len(self.vocab['diagnostic']))
        self.form_head = nn.Linear(feature_dim, len(self.vocab['form']))
        self.rhythm_head = nn.Linear(feature_dim, len(self.vocab['rhythm']))

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone.encode_features(x, pooling=self.pooling, quantize=self.quantize)
        features = self.norm(features)
        features = self.dropout(features)
        return {
            'diagnostic': self.diagnostic_head(features),
            'form': self.form_head(features),
            'rhythm': self.rhythm_head(features),
        }


class DiagnosisProvider:
    """Use the diagnosis refinement branch as the diagnostic branch for TRACE reports."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        tokenizer_checkpoint: str,
        device: str | torch.device,
        class_names: Sequence[str] | None = None,
        seq_length: int = 2250,
        patch_size: int = 75,
        latent_ratio: float = 0.1,
        channels: int = 12,
        codebook_size: int = 256,
        residual_levels: int = 2,
        pooling: str = 'flatten',
        dropout: float = 0.1,
        quantize: bool = False,
        threshold: float = 0.5,
        force_at_least_one: bool = True,
    ):
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.force_at_least_one = bool(force_at_least_one)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint_class_names = checkpoint.get('class_names')
        if checkpoint_class_names:
            resolved_class_names = [str(name) for name in checkpoint_class_names]
        elif class_names is not None:
            resolved_class_names = [str(name) for name in class_names]
        else:
            raise ValueError('class_names are required when the diagnosis checkpoint does not store them.')
        self.class_names = resolved_class_names

        self.model = TRACEDiagnosisClassifier(
            tokenizer_checkpoint=tokenizer_checkpoint,
            num_classes=len(self.class_names),
            seq_length=seq_length,
            patch_size=patch_size,
            latent_ratio=latent_ratio,
            channels=channels,
            codebook_size=codebook_size,
            residual_levels=residual_levels,
            pooling=pooling,
            dropout=dropout,
            quantize=quantize,
            freeze_backbone=True,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def predict_batch_label_dicts(
        self,
        signals: torch.Tensor,
        *,
        diagnostic_vocab: Sequence[str],
        use_amp: bool = False,
    ) -> list[Dict[str, float]]:
        vocab_set = set(diagnostic_vocab)
        signals = signals.to(self.device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = self.model(signals)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        outputs: list[Dict[str, float]] = []
        for row in probs:
            selected = {
                label: float(score * 100.0)
                for label, score in zip(self.class_names, row.tolist())
                if score >= self.threshold and label in vocab_set
            }
            if not selected and self.force_at_least_one and len(row) > 0:
                top_idx = int(np.argmax(row))
                top_label = self.class_names[top_idx]
                if top_label in vocab_set:
                    selected[top_label] = float(row[top_idx] * 100.0)
            outputs.append(selected)
        return outputs


def load_diagnostic_provider(
    checkpoint_path: str | Path,
    *,
    tokenizer_checkpoint: str,
    device: str | torch.device,
    dataset: str,
    seq_length: int = 2250,
    patch_size: int = 75,
    latent_ratio: float = 0.1,
    channels: int = 12,
    codebook_size: int = 256,
    residual_levels: int = 2,
    pooling: str = 'flatten',
    dropout: float = 0.1,
    quantize: bool = False,
    threshold: float = 0.5,
    force_at_least_one: bool = True,
) -> DiagnosisProvider:
    dataset_key = dataset.lower()
    if dataset_key in {'ptbxl', 'ptb-xl'}:
        class_names = PTBXL_SUPERCLASSES
    elif dataset_key in {'cpsc2018', 'cpsc'}:
        class_names = CPSC2018_CLASSES
    else:
        raise ValueError(f'Unsupported diagnostic dataset: {dataset}')
    return DiagnosisProvider(
        checkpoint_path=checkpoint_path,
        tokenizer_checkpoint=tokenizer_checkpoint,
        device=device,
        class_names=class_names,
        seq_length=seq_length,
        patch_size=patch_size,
        latent_ratio=latent_ratio,
        channels=channels,
        codebook_size=codebook_size,
        residual_levels=residual_levels,
        pooling=pooling,
        dropout=dropout,
        quantize=quantize,
        threshold=threshold,
        force_at_least_one=force_at_least_one,
    )


def build_control_loaders(
    datasets: Mapping[str, Dataset],
    batch_size: int,
    num_workers: int,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    return {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory),
        'valid': DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory),
    }


def compute_control_loss(logits: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor], criterion: nn.Module) -> torch.Tensor:
    loss = 0.0
    for group in CONTROL_GROUPS:
        if logits[group].numel() == 0:
            continue
        loss = loss + criterion(logits[group], targets[group])
    return loss


def _macro_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    scores = []
    for i in range(y_true.shape[1]):
        column = y_true[:, i]
        if np.unique(column).size < 2:
            continue
        scores.append(roc_auc_score(column, y_prob[:, i]))
    return float(np.mean(scores)) if scores else float('nan')


def compute_group_metrics(logits: np.ndarray, targets: np.ndarray, threshold: float) -> Dict[str, float]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    predictions = (probabilities >= threshold).astype(np.int64)
    targets = targets.astype(np.int64)
    return {
        'macro_f1': float(f1_score(targets, predictions, average='macro', zero_division=0)),
        'macro_auroc': _macro_auroc(targets, probabilities),
    }


def train_control_epoch(
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
    for signals, diagnostic, form, rhythm in tqdm(loader, desc='train', leave=False):
        signals = signals.to(device, non_blocking=True)
        targets = {
            'diagnostic': diagnostic.to(device, non_blocking=True),
            'form': form.to(device, non_blocking=True),
            'rhythm': rhythm.to(device, non_blocking=True),
        }
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            logits = model(signals)
            loss = compute_control_loss(logits, targets, criterion)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = signals.size(0)
        running_loss += loss.detach().item() * batch_size
        sample_count += batch_size
    return running_loss / max(sample_count, 1)


@torch.no_grad()
def evaluate_control_epoch(
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
    logits_buffer = {group: [] for group in CONTROL_GROUPS}
    targets_buffer = {group: [] for group in CONTROL_GROUPS}
    for signals, diagnostic, form, rhythm in tqdm(loader, desc='eval', leave=False):
        signals = signals.to(device, non_blocking=True)
        targets = {
            'diagnostic': diagnostic.to(device, non_blocking=True),
            'form': form.to(device, non_blocking=True),
            'rhythm': rhythm.to(device, non_blocking=True),
        }
        with autocast(enabled=use_amp):
            logits = model(signals)
            loss = compute_control_loss(logits, targets, criterion)
        batch_size = signals.size(0)
        running_loss += loss.detach().item() * batch_size
        sample_count += batch_size
        for group in CONTROL_GROUPS:
            logits_buffer[group].append(logits[group].detach().cpu())
            targets_buffer[group].append(targets[group].detach().cpu())

    metrics = {}
    group_logits_np = {}
    group_targets_np = {}
    group_scores = []
    for group in CONTROL_GROUPS:
        group_logits = torch.cat(logits_buffer[group], dim=0).numpy()
        group_targets = torch.cat(targets_buffer[group], dim=0).numpy()
        group_logits_np[group] = group_logits
        group_targets_np[group] = group_targets
        if group_logits.shape[1] == 0:
            metrics[f'{group}_macro_f1'] = float('nan')
            metrics[f'{group}_macro_auroc'] = float('nan')
            continue
        group_metrics = compute_group_metrics(group_logits, group_targets, threshold=threshold)
        metrics[f'{group}_macro_f1'] = group_metrics['macro_f1']
        metrics[f'{group}_macro_auroc'] = group_metrics['macro_auroc']
        if not np.isnan(group_metrics['macro_f1']):
            group_scores.append(group_metrics['macro_f1'])
    metrics['mean_macro_f1'] = float(np.mean(group_scores)) if group_scores else float('nan')
    return {
        'loss': running_loss / max(sample_count, 1),
        'metrics': metrics,
        'logits': group_logits_np,
        'targets': group_targets_np,
    }


def save_control_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    vocab: Mapping[str, Sequence[str]],
    metrics: Mapping[str, float],
    args_dict: Mapping[str, object],
    tokenizer_checkpoint: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'control_vocab': {group: list(vocab.get(group, [])) for group in CONTROL_GROUPS},
            'metrics': dict(metrics),
            'args': dict(args_dict),
            'tokenizer_checkpoint': tokenizer_checkpoint,
        },
        path,
    )


def load_control_checkpoint(path: str | Path, map_location: str = 'cpu') -> Dict[str, object]:
    return torch.load(path, map_location=map_location)


def load_report_control_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    tokenizer_checkpoint: str | None = None,
) -> tuple[ReportControlClassifier, Dict[str, list[str]], Dict[str, object]]:
    checkpoint = load_control_checkpoint(checkpoint_path, map_location='cpu')
    vocab = {group: list(checkpoint['control_vocab'].get(group, [])) for group in CONTROL_GROUPS}
    args = checkpoint.get('args', {})
    resolved_tokenizer_checkpoint = tokenizer_checkpoint or checkpoint.get('tokenizer_checkpoint', '')
    if not resolved_tokenizer_checkpoint:
        raise ValueError('Tokenizer checkpoint is required to reconstruct the TRACE control model.')

    model = ReportControlClassifier(
        tokenizer_checkpoint=resolved_tokenizer_checkpoint,
        vocab=vocab,
        seq_length=int(args.get('seq_length', 2250)),
        patch_size=int(args.get('patch_size', 75)),
        latent_ratio=float(args.get('latent_ratio', 0.1)),
        channels=int(args.get('channels', 12)),
        codebook_size=int(args.get('codebook_size', 256)),
        residual_levels=int(args.get('residual_levels', 2)),
        pooling=str(args.get('pooling', 'flatten')),
        dropout=float(args.get('dropout', 0.1)),
        quantize=bool(args.get('use_quantized_latents', False)),
        freeze_backbone=True,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, vocab, checkpoint


@torch.no_grad()
def predict_control_prompts(
    model: nn.Module,
    loader: DataLoader,
    *,
    vocab: Mapping[str, Sequence[str]],
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = False,
) -> list[str]:
    token_maps = build_control_token_maps(vocab)['groups']
    prompts = []
    model.eval()
    for batch in tqdm(loader, desc='predict_control', leave=False):
        if isinstance(batch, (tuple, list)):
            signals = batch[0]
        else:
            signals = batch
        signals = signals.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(signals)
        binary_predictions = {
            group: (torch.sigmoid(logits[group]).detach().cpu().numpy() >= threshold).astype(np.int64)
            for group in CONTROL_GROUPS
        }
        batch_size = signals.size(0)
        for row_idx in range(batch_size):
            prompt = render_control_prompt_from_binary(
                {group: binary_predictions[group][row_idx] for group in CONTROL_GROUPS},
                vocab=vocab,
                token_maps=token_maps,
            )
            prompts.append(prompt)
    return prompts


@torch.no_grad()
def predict_fused_control_prompts(
    *,
    diagnostic_provider: DiagnosisProvider,
    control_model: nn.Module,
    loader: DataLoader,
    vocab: Mapping[str, Sequence[str]],
    device: torch.device,
    control_threshold: float = 0.5,
    use_amp: bool = False,
) -> list[str]:
    """Fuse diagnostic predictions with form/rhythm predictions."""
    token_maps = build_control_token_maps(vocab)['groups']
    prompts = []
    control_model.eval()
    for batch in tqdm(loader, desc='predict_v_fused_control', leave=False):
        if isinstance(batch, (tuple, list)):
            signals = batch[0]
        else:
            signals = batch
        signals = signals.to(device, non_blocking=True)
        diagnostic_dicts = diagnostic_provider.predict_batch_label_dicts(
            signals,
            diagnostic_vocab=vocab.get('diagnostic', []),
            use_amp=use_amp,
        )
        with autocast(enabled=use_amp):
            control_logits = control_model(signals)
        form_rhythm_binary = {
            group: (torch.sigmoid(control_logits[group]).detach().cpu().numpy() >= control_threshold).astype(np.int64)
            for group in ('form', 'rhythm')
        }
        batch_size = signals.size(0)
        for row_idx in range(batch_size):
            label_dicts = {
                'diagnostic': diagnostic_dicts[row_idx],
                'form': {
                    label: 100.0
                    for label, flag in zip(vocab.get('form', []), form_rhythm_binary['form'][row_idx].tolist())
                    if flag
                },
                'rhythm': {
                    label: 100.0
                    for label, flag in zip(vocab.get('rhythm', []), form_rhythm_binary['rhythm'][row_idx].tolist())
                    if flag
                },
            }
            prompts.append(render_control_prompt(label_dicts, vocab=vocab, token_maps=token_maps))
    return prompts
