
import argparse
from pathlib import Path

from train_diagnosis_ckpt_ningbo import (
    DEFAULT_NINGBO_LABEL_NAMES_JSON,
    DEFAULT_NINGBO_ROOT,
    NingboDownstreamDataset,
    infer_ningbo_label_dim,
    load_ningbo_label_names,
)
from utils.downstream_trace import add_common_args, run_downstream_experiment


def parse_args():
    parser = argparse.ArgumentParser("Ningbo downstream diagnosis evaluation with the diagnosis refinement branch")
    add_common_args(
        parser,
        default_tokenizer_checkpoint="ckpt/ningbo/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_Ningbo_st2_best.pth",
        default_output_dir="downstream/ningbo_trace",
    )
    parser.set_defaults(
        downstream_checkpoint="ckpt/ningbo/downstream_refined/best.pt",
        eval_only=True,
    )
    parser.add_argument("--ningbo-root", type=str, default=DEFAULT_NINGBO_ROOT)
    parser.add_argument("--train-index", type=str, default="train_index.csv")
    parser.add_argument("--valid-index", type=str, default="val_index.csv")
    parser.add_argument("--test-index", type=str, default="test_index.csv")
    parser.add_argument("--label-names-json", type=str, default=DEFAULT_NINGBO_LABEL_NAMES_JSON)
    parser.add_argument("--source-rate", type=int, default=500)
    parser.add_argument("--target-rate", type=int, default=250)
    parser.add_argument("--normalize", action="store_true", default=True)
    parser.add_argument("--no-normalize", action="store_false", dest="normalize")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def build_ningbo_datasets(args):
    root = Path(args.ningbo_root)
    train_index = root / args.train_index
    valid_index = root / args.valid_index
    test_index = root / args.test_index

    label_dim = infer_ningbo_label_dim(train_index)
    class_names = load_ningbo_label_names(args.label_names_json, label_dim)
    common_kwargs = {
        "data_dir": root / "data",
        "label_names": class_names,
        "seq_length": args.seq_length,
        "latent_ratio": args.latent_ratio,
        "source_rate": args.source_rate,
        "target_rate": args.target_rate,
        "channels": args.channels,
        "normalize": args.normalize,
    }
    datasets = {
        "train": NingboDownstreamDataset(train_index, max_samples=args.max_train_samples, **common_kwargs),
        "valid": NingboDownstreamDataset(valid_index, max_samples=args.max_valid_samples, **common_kwargs),
        "test": NingboDownstreamDataset(test_index, max_samples=args.max_test_samples, **common_kwargs),
    }
    return datasets, class_names


def main():
    args = parse_args()
    datasets, class_names = build_ningbo_datasets(args)
    run_downstream_experiment(args, datasets, class_names, task_name="ningbo_diagnosis")


if __name__ == "__main__":
    main()
