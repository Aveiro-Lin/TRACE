
import argparse

from utils.downstream_trace import add_common_args, build_cpsc2018_datasets, run_downstream_experiment


def parse_args():
    parser = argparse.ArgumentParser('CPSC2018 downstream classification with the diagnosis refinement branch on top of the original baseline')
    add_common_args(
        parser,
        default_tokenizer_checkpoint='ckpt/cpsc2018/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_cpsc2018_st2.pth',
        default_output_dir='downstream/cpsc2018_trace',
    )
    parser.set_defaults(base_downstream_checkpoint='ckpt/cpsc2018/downstream_base/best.pt')
    parser.add_argument('--train-reference-csv', type=str, default='data/TRAIN_reference_cpsc2018.csv')
    parser.add_argument('--valid-reference-csv', type=str, default='data/VAL_reference_cpsc2018.csv')
    parser.add_argument('--test-reference-csv', type=str, default='data/TEST_reference_cpsc2018.csv')
    parser.add_argument('--train-signals-npy', type=str, default='data/records2250_cpsc2018_train.npy')
    parser.add_argument('--valid-signals-npy', type=str, default='data/records2250_cpsc2018_val.npy')
    parser.add_argument('--test-signals-npy', type=str, default='data/records2250_cpsc2018_test.npy')
    return parser.parse_args()


def main():
    args = parse_args()
    datasets, class_names = build_cpsc2018_datasets(
        train_reference_csv=args.train_reference_csv,
        valid_reference_csv=args.valid_reference_csv,
        test_reference_csv=args.test_reference_csv,
        train_signals_npy=args.train_signals_npy,
        valid_signals_npy=args.valid_signals_npy,
        test_signals_npy=args.test_signals_npy,
        seq_length=args.seq_length,
    )
    run_downstream_experiment(args, datasets, class_names, task_name='cpsc2018_diagnosis')


if __name__ == '__main__':
    main()
