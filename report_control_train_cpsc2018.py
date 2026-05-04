import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler

from utils.cpsc2018_utils import CPSC2018Utils
from utils.report_controls import (
    ReportControlClassifier,
    ReportControlDataset,
    build_control_loaders,
    build_control_targets,
    build_control_vocab_from_dataframe,
    evaluate_control_epoch,
    extract_signal_window,
    save_control_checkpoint,
    save_control_vocab,
    train_control_epoch,
)


def parse_args():
    parser = argparse.ArgumentParser('Train the CPSC2018 report-control classifier')
    parser.add_argument('--tokenizer-checkpoint', type=str, default='ckpt/cpsc2018/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_cpsc2018_st2.pth')
    parser.add_argument('--output-dir', type=str, default='ckpt/cpsc2018/report_control')
    parser.add_argument('--vocab-path', type=str, default='ckpt/cpsc2018/report_control/report_control_vocab_cpsc2018.json')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--base-dir', type=str, default='data/cpsc2018/Training_WFDB')
    parser.add_argument('--train-reference-csv', type=str, default='data/TRAIN_reference_cpsc2018.csv')
    parser.add_argument('--valid-reference-csv', type=str, default='data/VAL_reference_cpsc2018.csv')
    parser.add_argument('--test-reference-csv', type=str, default='data/TEST_reference_cpsc2018.csv')
    parser.add_argument('--train-signals-npy', type=str, default='data/records2250_cpsc2018_train.npy')
    parser.add_argument('--valid-signals-npy', type=str, default='data/records2250_cpsc2018_val.npy')
    parser.add_argument('--test-signals-npy', type=str, default='data/records2250_cpsc2018_test.npy')
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
    parser.add_argument('--pin-memory', action='store_true')
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


def build_datasets(args):
    cpsc = CPSC2018Utils(args.base_dir)
    train_df = cpsc.load_labels_cpsc2018(args.train_reference_csv)
    valid_df = cpsc.load_labels_cpsc2018(args.valid_reference_csv)
    test_df = cpsc.load_labels_cpsc2018(args.test_reference_csv)

    vocab = build_control_vocab_from_dataframe(train_df)
    save_control_vocab(args.vocab_path, vocab)

    train_targets = build_control_targets(train_df, vocab)
    valid_targets = build_control_targets(valid_df, vocab)
    test_targets = build_control_targets(test_df, vocab)

    train_signals = extract_signal_window(np.load(args.train_signals_npy, mmap_mode='r'), seq_length=args.seq_length, latent_ratio=args.latent_ratio)
    valid_signals = extract_signal_window(np.load(args.valid_signals_npy, mmap_mode='r'), seq_length=args.seq_length, latent_ratio=args.latent_ratio)
    test_signals = extract_signal_window(np.load(args.test_signals_npy, mmap_mode='r'), seq_length=args.seq_length, latent_ratio=args.latent_ratio)

    datasets = {
        'train': ReportControlDataset(train_signals, train_targets['diagnostic'], train_targets['form'], train_targets['rhythm']),
        'valid': ReportControlDataset(valid_signals, valid_targets['diagnostic'], valid_targets['form'], valid_targets['rhythm']),
        'test': ReportControlDataset(test_signals, test_targets['diagnostic'], test_targets['form'], test_targets['rhythm']),
    }
    return datasets, vocab


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets, vocab = build_datasets(args)
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
        save_control_checkpoint(output_dir / 'last.pt', model, optimizer, epoch, vocab, metrics_payload, vars(args), args.tokenizer_checkpoint)
        if metric_value >= best_value:
            best_value = metric_value
            save_control_checkpoint(output_dir / 'best.pt', model, optimizer, epoch, vocab, metrics_payload, vars(args), args.tokenizer_checkpoint)

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
