
import argparse

from utils.downstream_trace import add_common_args, build_ptbxl_datasets, run_downstream_experiment


def parse_args():
    parser = argparse.ArgumentParser('PTB-XL downstream classification with the diagnosis refinement branch on top of the original baseline')
    add_common_args(
        parser,
        default_tokenizer_checkpoint='ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth',
        default_output_dir='downstream/ptbxl_trace',
    )
    parser.set_defaults(base_downstream_checkpoint='ckpt/ptbxl/downstream_base/best.pt')
    parser.add_argument('--metadata-csv', type=str, default='data/ptbxl_database.csv')
    parser.add_argument('--scp-statements-csv', type=str, default='data/scp_statements.csv')
    parser.add_argument('--signals-npy', type=str, default='data/records2250.npy')
    parser.add_argument('--valid-fold', type=int, default=9)
    parser.add_argument('--test-fold', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets, class_names = build_ptbxl_datasets(
        metadata_csv=args.metadata_csv,
        scp_statements_csv=args.scp_statements_csv,
        signals_npy=args.signals_npy,
        seq_length=args.seq_length,
        latent_ratio=args.latent_ratio,
        valid_fold=args.valid_fold,
        test_fold=args.test_fold,
    )
    run_downstream_experiment(args, datasets, class_names, task_name='ptbxl_superdiagnostic')


if __name__ == '__main__':
    main()
