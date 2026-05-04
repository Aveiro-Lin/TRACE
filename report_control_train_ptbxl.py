import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler

from utils.report_controls import (
    ReportControlClassifier,
    ReportControlDataset,
    add_ptbxl_diagnostic_superclass_column,
    build_control_loaders,
    build_control_targets,
    build_ptbxl_v15_vocab,
    compute_ptbxl_label_aggregations,
    evaluate_control_epoch,
    extract_signal_window,
    handle_ptbxl_sex,
    save_control_checkpoint,
    save_control_vocab,
    train_control_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser('Train the PTB-XL report-control classifier')
    parser.add_argument('--tokenizer-checkpoint', type=str, default='ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth')
    parser.add_argument('--output-dir', type=str, default='ckpt/ptbxl/report_control')
    parser.add_argument('--vocab-path', type=str, default='ckpt/ptbxl/report_control/report_control_vocab_ptbxl.json')
    parser.add_argument('--metadata-csv', type=str, default='data/ptbxl_database.csv')
    parser.add_argument('--scp-statements-csv', type=str, default='data/scp_statements.csv')
    parser.add_argument('--signals-npy', type=str, default='data/full_data.npy')
    parser.add_argument('--valid-fold', type=int, default=9)
    parser.add_argument('--test-fold', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pooling', choices=['mean', 'flatten'], default='flatten')
    parser.add_argument('--use-quantized-latents', action='store_true')
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--seq-length', type=int, default=2250)
    parser.add_argument('--patch-size', type=int, default=75)
    parser.add_argument('--latent-ratio', type=float, default=0.1)
    parser.add_argument('--channels', type=int, default=12)
    parser.add_argument('--codebook-size', type=int, default=256)
    parser.add_argument('--residual-levels', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--target-metric', type=str, default='mean_macro_f1')
    parser.add_argument('--min-form-count', type=int, default=300)
    parser.add_argument('--min-rhythm-count', type=int, default=100)
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_datasets(args):
    metadata = pd.read_csv(args.metadata_csv, index_col='ecg_id')
    metadata.scp_codes = metadata.scp_codes.apply(ast.literal_eval)
    metadata = compute_ptbxl_label_aggregations(metadata, args.scp_statements_csv)
    metadata = add_ptbxl_diagnostic_superclass_column(metadata, args.scp_statements_csv)
    metadata.sex = metadata.sex.apply(handle_ptbxl_sex)

    signals = np.load(args.signals_npy, mmap_mode='r')
    signals = extract_signal_window(signals, seq_length=args.seq_length, latent_ratio=args.latent_ratio)

    train_mask = (metadata['strat_fold'] != args.valid_fold) & (metadata['strat_fold'] != args.test_fold)
    valid_mask = metadata['strat_fold'] == args.valid_fold
    test_mask = metadata['strat_fold'] == args.test_fold

    train_df = metadata[train_mask].copy()
    valid_df = metadata[valid_mask].copy()
    test_df = metadata[test_mask].copy()

    column_map = {'diagnostic': 'diagnostic_superclass', 'form': 'form', 'rhythm': 'rhythm'}
    vocab = build_ptbxl_v15_vocab(
        train_df,
        min_form_count=args.min_form_count,
        min_rhythm_count=args.min_rhythm_count,
    )
    save_control_vocab(args.vocab_path, vocab)

    all_targets = build_control_targets(metadata, vocab, column_map=column_map)
    train_idx = np.where(train_mask.to_numpy())[0]
    valid_idx = np.where(valid_mask.to_numpy())[0]
    test_idx = np.where(test_mask.to_numpy())[0]

    datasets = {
        'train': ReportControlDataset(signals[train_idx], all_targets['diagnostic'][train_idx], all_targets['form'][train_idx], all_targets['rhythm'][train_idx]),
        'valid': ReportControlDataset(signals[valid_idx], all_targets['diagnostic'][valid_idx], all_targets['form'][valid_idx], all_targets['rhythm'][valid_idx]),
        'test': ReportControlDataset(signals[test_idx], all_targets['diagnostic'][test_idx], all_targets['form'][test_idx], all_targets['rhythm'][test_idx]),
    }
    return datasets, vocab, column_map


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets, vocab, column_map = build_datasets(args)
    loaders = build_control_loaders(datasets, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=args.pin_memory)

    device = torch.device(args.device)
    use_amp = device.type == 'cuda' and not args.disable_amp
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
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=use_amp)
    history_path = output_dir / 'history.jsonl'

    print(
        json.dumps(
            {
                'tokenizer_checkpoint': args.tokenizer_checkpoint,
                'output_dir': str(output_dir),
                'vocab_path': args.vocab_path,
                'control_column_map': column_map,
                'vocab_sizes': {group: len(vocab.get(group, [])) for group in vocab.keys()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    best_value = -float('inf')
    for epoch in range(args.epochs):
        train_loss = train_control_epoch(model, loaders['train'], optimizer, criterion, device, scaler, use_amp)
        valid_result = evaluate_control_epoch(model, loaders['valid'], criterion, device, args.threshold, use_amp)
        valid_metrics = valid_result['metrics']
        metric_value = float(valid_metrics.get(args.target_metric, -float('inf')))
        log_row = {'epoch': epoch, 'train_loss': train_loss, 'valid_loss': valid_result['loss'], **valid_metrics}
        with history_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(log_row, ensure_ascii=False) + '\n')
        print(json.dumps(log_row, ensure_ascii=False), flush=True)

        metrics_payload = {'loss': valid_result['loss'], **valid_metrics}
        args_dict = vars(args).copy()
        args_dict['control_column_map'] = column_map
        save_control_checkpoint(output_dir / 'last.pt', model, optimizer, epoch, vocab, metrics_payload, args_dict, args.tokenizer_checkpoint)
        if metric_value >= best_value:
            best_value = metric_value
            save_control_checkpoint(output_dir / 'best.pt', model, optimizer, epoch, vocab, metrics_payload, args_dict, args.tokenizer_checkpoint)

    best_checkpoint = torch.load(output_dir / 'best.pt', map_location='cpu')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test_result = evaluate_control_epoch(model, loaders['test'], criterion, device, args.threshold, use_amp)
    payload = {
        'checkpoint': str(output_dir / 'best.pt'),
        'vocab_path': args.vocab_path,
        'best_valid_metrics': best_checkpoint.get('metrics', {}),
        'test_loss': test_result['loss'],
        **test_result['metrics'],
    }
    with (output_dir / 'test_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
