import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler

from utils.downstream_trace import (
    TRACEDiagnosisClassifier,
    build_cpsc2018_datasets,
    build_dataloaders,
    build_ptbxl_datasets,
    evaluate,
    save_checkpoint,
    train_one_epoch,
)


def parse_args():
    p = argparse.ArgumentParser('Train residual-booster downstream checkpoints on top of the original ST2.0 downstream baseline')
    p.add_argument('--dataset', choices=['ptbxl', 'cpsc2018'], required=True)
    p.add_argument('--tokenizer-checkpoint', type=str, default='')
    p.add_argument('--base-downstream-checkpoint', type=str, default='')
    p.add_argument('--output-dir', type=str, default='')
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
    p.add_argument('--metadata-csv', type=str, default='data/ptbxl_database.csv')
    p.add_argument('--scp-statements-csv', type=str, default='data/scp_statements.csv')
    p.add_argument('--signals-npy', type=str, default='data/records2250.npy')
    p.add_argument('--valid-fold', type=int, default=9)
    p.add_argument('--test-fold', type=int, default=10)
    p.add_argument('--train-reference-csv', type=str, default='data/TRAIN_reference_cpsc2018.csv')
    p.add_argument('--valid-reference-csv', type=str, default='data/VAL_reference_cpsc2018.csv')
    p.add_argument('--test-reference-csv', type=str, default='data/TEST_reference_cpsc2018.csv')
    p.add_argument('--train-signals-npy', type=str, default='data/records2250_cpsc2018_train.npy')
    p.add_argument('--valid-signals-npy', type=str, default='data/records2250_cpsc2018_val.npy')
    p.add_argument('--test-signals-npy', type=str, default='data/records2250_cpsc2018_test.npy')
    return p.parse_args()


def apply_dataset_defaults(args):
    if args.dataset == 'ptbxl':
        if not args.tokenizer_checkpoint:
            args.tokenizer_checkpoint = 'ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth'
        if not args.base_downstream_checkpoint:
            args.base_downstream_checkpoint = 'ckpt/ptbxl/downstream_base/best.pt'
        if not args.output_dir:
            args.output_dir = 'ckpt/ptbxl/downstream_refined'
    else:
        if not args.tokenizer_checkpoint:
            args.tokenizer_checkpoint = 'ckpt/cpsc2018/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_cpsc2018_st2.pth'
        if not args.base_downstream_checkpoint:
            args.base_downstream_checkpoint = 'ckpt/cpsc2018/downstream_base/best.pt'
        if not args.output_dir:
            args.output_dir = 'ckpt/cpsc2018/downstream_refined'
    return args


def validate_dataset_routing(args):
    name = Path(args.tokenizer_checkpoint).name.lower()
    if args.dataset == 'cpsc2018' and 'cpsc2018' not in name:
        raise ValueError(f'CPSC2018 downstream ckpt must use a CPSC2018 tokenizer checkpoint, got: {args.tokenizer_checkpoint}')
    if args.dataset == 'ptbxl' and 'cpsc2018' in name:
        raise ValueError(f'PTB-XL downstream ckpt must not use a CPSC2018 tokenizer checkpoint, got: {args.tokenizer_checkpoint}')
    return args


def build_datasets(args):
    if args.dataset == 'ptbxl':
        return build_ptbxl_datasets(
            metadata_csv=args.metadata_csv,
            scp_statements_csv=args.scp_statements_csv,
            signals_npy=args.signals_npy,
            seq_length=args.seq_length,
            latent_ratio=args.latent_ratio,
            valid_fold=args.valid_fold,
            test_fold=args.test_fold,
        )
    return build_cpsc2018_datasets(
        train_reference_csv=args.train_reference_csv,
        valid_reference_csv=args.valid_reference_csv,
        test_reference_csv=args.test_reference_csv,
        train_signals_npy=args.train_signals_npy,
        valid_signals_npy=args.valid_signals_npy,
        test_signals_npy=args.test_signals_npy,
        seq_length=args.seq_length,
    )


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
                'train_base_branch': args.train_base_branch,
                'train_backbone_with_booster': args.train_backbone_with_booster,
                'residual_strategy': 'base_logits + residual_scale * delta_logits',
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    datasets, class_names = build_datasets(args)
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
    }
    with (output_dir / 'ckpt_summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
