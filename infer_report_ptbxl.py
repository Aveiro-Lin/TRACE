# coding: utf-8

"""Autoregressive PTB-XL report generation for the TRACE report head.

This replaces the old teacher-forced/logit-argmax evaluation path.  The model is
now evaluated like ECG report generation baselines such as MEIT: construct a
prompt from ECG/control tokens, call ``model.generate()``, decode only newly
generated assistant tokens, and then parse/report diagnostic consistency.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import random
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers.cache_utils import DynamicCache
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from utils.my_templates import Reports
from utils.trace_tokenizer import Tokenizer
from utils.report_controls import (
    add_ptbxl_diagnostic_superclass_column,
    build_control_token_maps,
    load_report_control_model,
    load_diagnostic_provider,
    predict_fused_control_prompts,
)


os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


SAMPLING_RATE = 500
SEQ_LENGTH = 2250
PATCH_SIZE = 75
LATENT_RATIO = 0.1
CHANNELS = 12
CODEBOOK_SIZE = 256
RESIDUAL_LEVELS = 2
VALID_FOLD = 9
TEST_FOLD = 10
DIAGNOSIS_THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent
TOKENIZER_DIR = (
    f"level_{RESIDUAL_LEVELS}_code_{CODEBOOK_SIZE}_len_{SEQ_LENGTH}_ratio_{LATENT_RATIO}"
)
ECG_TOKENIZER_PATH = PROJECT_ROOT / f"ckpt/ptbxl/tokenizer/ecg_tokenizer_{TOKENIZER_DIR}_st2.pth"
DEFAULT_MERGED_MODEL_PATH = PROJECT_ROOT / "ckpt/ptbxl/report_merged_model"
DEFAULT_CONTROL_CHECKPOINT_PATH = PROJECT_ROOT / "ckpt/ptbxl/report_control/best.pt"
DEFAULT_DIAGNOSIS_CHECKPOINT_PATH = PROJECT_ROOT / "ckpt/ptbxl/downstream_refined/best.pt"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "data/generated_ptbxl_st2"
DEFAULT_CONTROL_CACHE_PATH = PROJECT_ROOT / "data/control_prompts_ptbxl_st2_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean PTB-XL TRACE reports with model.generate()."
    )
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--token_batch_size", type=int, default=64)
    parser.add_argument("--control_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--max_prompt_length", type=int, default=3072)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--merged_model_path", type=Path, default=DEFAULT_MERGED_MODEL_PATH)
    parser.add_argument("--control_checkpoint_path", type=Path, default=DEFAULT_CONTROL_CHECKPOINT_PATH)
    parser.add_argument("--diagnosis-checkpoint-path", type=Path, default=DEFAULT_DIAGNOSIS_CHECKPOINT_PATH)
    parser.add_argument("--ecg_tokenizer_path", type=Path, default=ECG_TOKENIZER_PATH)
    parser.add_argument("--output_prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--control_cache_path", type=Path, default=DEFAULT_CONTROL_CACHE_PATH)
    parser.add_argument("--refresh_control_cache", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def patch_phi3_dynamic_cache_compat() -> None:
    """Keep saved Phi-3 remote code compatible with newer transformers.

    The merged TRACE model directory contains the original Phi-3 modeling file,
    whose generation path calls ``DynamicCache.get_max_length()``.  In newer
    transformers versions this method was renamed to ``get_max_cache_shape()``.
    Adding the alias lets us keep fast cached autoregressive generation.
    """

    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = DynamicCache.get_max_cache_shape


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_text_tokenizer(model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = "left"
    return tokenizer


def load_generation_model(model_path: Path):
    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        "trust_remote_code": True,
        "attn_implementation": "eager",
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))
    model.eval()
    return model


def first_model_device(model: torch.nn.Module) -> torch.device:
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_ecg_tokenizer(checkpoint_path: Path, device: torch.device) -> Tokenizer:
    vq_kwargs = {"residual_levels": RESIDUAL_LEVELS}
    tokenizer = Tokenizer(
        seq_length=SEQ_LENGTH,
        patch_size=PATCH_SIZE,
        latent_ratio=LATENT_RATIO,
        channels=CHANNELS,
        codebook_size=CODEBOOK_SIZE,
        vq_kwargs=vq_kwargs,
    ).to(device)
    weights = torch.load(checkpoint_path, weights_only=False, map_location=device)
    tokenizer.load_state_dict(weights["model_state_dict"])
    tokenizer.eval()
    return tokenizer


def compute_label_aggregations(df: pd.DataFrame, folder: Path) -> pd.DataFrame:
    df = df.copy()
    df["scp_codes_len"] = df.scp_codes.apply(lambda x: len(x))
    symptoms = ["diagnostic", "form", "rhythm"]
    aggregation_df = pd.read_csv(folder / "scp_statements.csv", index_col=0)
    agg_df = {symptom: aggregation_df[aggregation_df[symptom] == 1.0] for symptom in symptoms}

    def aggregate(symptom: str):
        def _aggregate(y_dic):
            tmp = {}
            for key in y_dic.keys():
                if key in agg_df[symptom].index and str(key) != "nan":
                    tmp.update([(key, y_dic[key])])
            return str(tmp)

        return _aggregate

    for symptom in symptoms:
        df[symptom] = df.scp_codes.apply(aggregate(symptom=symptom))
    return df


def handle_sex(sex):
    if isinstance(sex, str):
        return sex
    if isinstance(sex, (int, np.integer)):
        return "male" if sex == 1 else "female"
    if isinstance(sex, (float, np.floating)) and not np.isnan(sex):
        return "male" if int(sex) == 1 else "female"
    return "unknown"


def load_ptbxl_dataframe() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str], list[str], np.ndarray]:
    data_dir = PROJECT_ROOT / "data"
    combined_df = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    combined_df.scp_codes = combined_df.scp_codes.apply(ast.literal_eval)
    combined_df = compute_label_aggregations(combined_df, data_dir)
    combined_df = add_ptbxl_diagnostic_superclass_column(
        combined_df, str(data_dir / "scp_statements.csv")
    )
    combined_df.sex = combined_df.sex.apply(handle_sex)

    aggregation_df = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)
    diagnostic_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
    code_to_superclass = {
        str(code): str(diagnostic_agg_df.loc[code, "diagnostic_class"])
        for code in diagnostic_agg_df.index
        if isinstance(diagnostic_agg_df.loc[code, "diagnostic_class"], str)
    }

    names = combined_df["diagnostic"].drop_duplicates().apply(
        lambda x: list(ast.literal_eval(x).keys())
    )
    names = np.concatenate(names.values)
    names = np.unique(names)

    def get_diagnostic_class(name: str) -> str:
        if name == "NORMALECG":
            return "NORM"
        return diagnostic_agg_df[diagnostic_agg_df.index == name].diagnostic_class.tolist()[0]

    diagnostic_names = np.array([get_diagnostic_class(name) for name in names])
    diagnostic_names = np.unique(diagnostic_names)
    return combined_df, diagnostic_agg_df, code_to_superclass, names.tolist(), diagnostic_names


def load_signal() -> np.ndarray:
    x = np.load(PROJECT_ROOT / "data/full_data.npy")
    total_length = int(SEQ_LENGTH * (1 + LATENT_RATIO))
    start = int((x.shape[2] - total_length) / 2)
    return x[:, :, start : start + SEQ_LENGTH]


def select_split(
    combined_df: pd.DataFrame,
    signal: np.ndarray,
    split: str,
    max_samples: int | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    if split == "test":
        mask = combined_df.strat_fold == TEST_FOLD
    elif split == "valid":
        mask = combined_df.strat_fold == VALID_FOLD
    elif split == "train":
        mask = (combined_df.strat_fold != VALID_FOLD) & (combined_df.strat_fold != TEST_FOLD)
    else:
        mask = pd.Series(True, index=combined_df.index)

    positions = np.flatnonzero(mask.to_numpy())
    selected_df = combined_df.iloc[positions].copy()
    selected_signal = signal[positions]
    if max_samples is not None:
        selected_df = selected_df.iloc[:max_samples].copy()
        selected_signal = selected_signal[:max_samples]
    return selected_df, selected_signal


def tokenize_signals(
    signal: np.ndarray,
    ecg_tokenizer: Tokenizer,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list:
    dataset = TensorDataset(torch.tensor(signal, dtype=torch.float32))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    tokens = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Tokenizing ECG"):
            x = batch[0].to(device, non_blocking=True)
            ecg_tokens = ecg_tokenizer.tokenize(x).detach().cpu().tolist()
            tokens.extend(ecg_tokens)
    return tokens


def load_or_predict_control_prompts(
    selected_signal: np.ndarray,
    expected_count: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    if (
        args.control_cache_path
        and args.control_cache_path.exists()
        and not args.refresh_control_cache
    ):
        cached = json.loads(args.control_cache_path.read_text(encoding="utf-8"))
        if len(cached) == expected_count:
            print(f"Loaded cached control prompts: {args.control_cache_path}")
            return [str(item) for item in cached]
        print(
            "Ignoring control cache because length does not match "
            f"({len(cached)} != {expected_count})."
        )

    control_model, control_vocab, control_checkpoint = load_report_control_model(
        str(args.control_checkpoint_path),
        device=device,
        tokenizer_checkpoint=str(args.ecg_tokenizer_path),
    )
    build_control_token_maps(control_vocab)

    if not args.diagnosis_checkpoint_path.exists():
        raise FileNotFoundError(
            f"TRACE requires the diagnosis checkpoint: {args.diagnosis_checkpoint_path}"
        )
    diagnostic_provider = load_diagnostic_provider(
        str(args.diagnosis_checkpoint_path),
        tokenizer_checkpoint=str(args.ecg_tokenizer_path),
        device=device,
        dataset="ptbxl",
        seq_length=SEQ_LENGTH,
        patch_size=PATCH_SIZE,
        latent_ratio=LATENT_RATIO,
        channels=CHANNELS,
        codebook_size=CODEBOOK_SIZE,
        residual_levels=RESIDUAL_LEVELS,
        threshold=DIAGNOSIS_THRESHOLD,
        force_at_least_one=True,
    )

    loader = DataLoader(
        TensorDataset(torch.tensor(selected_signal, dtype=torch.float32)),
        batch_size=args.control_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    control_threshold = float(control_checkpoint.get("args", {}).get("threshold", 0.5))
    prompts = predict_fused_control_prompts(
        diagnostic_provider=diagnostic_provider,
        control_model=control_model,
        loader=loader,
        vocab=control_vocab,
        device=device,
        control_threshold=control_threshold,
        use_amp=device.type == "cuda",
    )

    del control_model
    del diagnostic_provider
    cleanup_cuda()

    args.control_cache_path.parent.mkdir(parents=True, exist_ok=True)
    args.control_cache_path.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote control prompt cache: {args.control_cache_path}")
    return prompts


def literal_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    return ast.literal_eval(value)


def sanitize_template_text(text: str) -> str:
    """Remove known notebook-template artifacts without touching shared templates."""

    return re.sub(r"<built-in method capitalize of str object at 0x[0-9a-fA-F]+>", "Provide", text)


def build_report_examples(selected_df: pd.DataFrame, text_tokenizer) -> list[dict[str, Any]]:
    examples = []
    for ecg_id, row in selected_df.iterrows():
        ecg_text = [f"<|ecg_{row.tokens[i][0] + 1}|>" for i in range(len(row.tokens))]
        ecg_input = "<|ecg_start|> " + " ".join(ecg_text) + " <|ecg_end|>"
        conditioned_ecg_input = f"{row.control_prompt} {ecg_input}"
        age = int(row.age) if not pd.isna(row.age) else 0
        sex = str(row.sex)
        symptom_dict_list = [
            literal_dict(row["diagnostic_superclass"]),
            literal_dict(row["form"]),
            literal_dict(row["rhythm"]),
        ]

        template = Reports.get_template(age, sex, conditioned_ecg_input, symptom_dict_list)[0]
        user = {"content": sanitize_template_text(f"{template[0]}\n{template[1]}"), "role": "user"}
        assistant = {"content": sanitize_template_text(template[2]), "role": "assistant"}
        prompt_text = text_tokenizer.apply_chat_template(
            [user],
            add_generation_prompt=True,
            tokenize=False,
        )

        examples.append(
            {
                "ecg_id": int(ecg_id),
                "strat_fold": int(row.strat_fold),
                "age": age,
                "sex": sex,
                "ptbxl_statement": str(row.report),
                "control_prompt": str(row.control_prompt),
                "prompt": prompt_text,
                "user_content": user["content"],
                "reference_report": assistant["content"],
            }
        )
    return examples


def generation_eos_ids(tokenizer) -> list[int]:
    eos_ids = set()
    for token in [tokenizer.eos_token, "<|end|>", "<|endoftext|>"]:
        if not token:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id >= 0 and token_id != tokenizer.unk_token_id:
            eos_ids.add(int(token_id))
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    return sorted(eos_ids)


def truncate_at_first_marker(text: str, markers: Iterable[str]) -> str:
    first_index = None
    for marker in markers:
        if not marker:
            continue
        index = text.find(marker)
        if index >= 0 and (first_index is None or index < first_index):
            first_index = index
    if first_index is None:
        return text
    return text[:first_index]


def clean_generated_report(text: str, tokenizer) -> str:
    text = text.replace(tokenizer.pad_token or "", "")
    text = text.strip()
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    text = truncate_at_first_marker(text, ["<|end|>", "</s>", tokenizer.eos_token])
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_diagnostic_class_from_code(
    name: str,
    diagnostic_agg_df: pd.DataFrame,
) -> str | None:
    if name == "NORMALECG":
        return "NORM"
    if name in diagnostic_agg_df.index:
        return diagnostic_agg_df[diagnostic_agg_df.index == name].diagnostic_class.tolist()[0]
    return None


def parse_superclass_labels(
    text: str,
    code_to_superclass: dict[str, str],
    superclass_names: list[str],
) -> list[str]:
    upper_text = text.upper()
    labels = set()
    direct_patterns = {
        "NORM": [r"\bNORM\b", r"NORMAL\s*ECG", r"NORMALECG"],
        "MI": [r"\bMI\b", r"MYOCARDIAL\s+INFARCTION", r"\bINFARCT"],
        "STTC": [r"\bSTTC\b", r"ST[-\s]?T", r"ISCHEM"],
        "CD": [r"\bCD\b", r"CONDUCTION\s+DISTURBANCE", r"BLOCK"],
        "HYP": [r"\bHYP\b", r"HYPERTROPHY"],
    }
    for label, patterns in direct_patterns.items():
        if any(re.search(pattern, upper_text) for pattern in patterns):
            labels.add(label)

    for code in sorted(code_to_superclass.keys(), key=len, reverse=True):
        if code and re.search(rf"(?<![A-Z0-9]){re.escape(code)}(?![A-Z0-9])", upper_text):
            labels.add(code_to_superclass[code])

    return [label for label in superclass_names if label in labels]


def resolve_diagnostic_labels(
    text: str,
    names: list[str],
    diagnostic_agg_df: pd.DataFrame,
    code_to_superclass: dict[str, str],
    superclass_names: list[str],
) -> list[str]:
    match = re.search(
        r"(?:#+\s*)?Diagnostic\s+Recommendation\s*(.*?)(?:<\|end\|>|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    extracted = match.group(1).strip() if match else text

    parsed_superclasses = parse_superclass_labels(
        extracted,
        code_to_superclass=code_to_superclass,
        superclass_names=superclass_names,
    )
    if parsed_superclasses:
        return parsed_superclasses

    upper_text = extracted.upper()
    handled = []
    for name in sorted(names, key=len, reverse=True):
        if name and name in upper_text:
            diagnostic_class = get_diagnostic_class_from_code(name, diagnostic_agg_df)
            if diagnostic_class is not None:
                handled.append(diagnostic_class)
    return sorted(set(handled), key=superclass_names.index)


def compute_label_onehot(
    label_names_arr: list[list[str]],
    diagnostic_names: np.ndarray,
) -> torch.Tensor:
    label_onehot_arr = torch.zeros((len(label_names_arr), len(diagnostic_names))).to(torch.bool)
    diagnostic_name_list = list(diagnostic_names)
    for i, label_names in enumerate(label_names_arr):
        for label_name in label_names:
            if label_name not in diagnostic_name_list:
                continue
            index = diagnostic_name_list.index(label_name)
            label_onehot_arr[i, index] = True
    return label_onehot_arr


def tensor_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def compute_macro_metrics_stable(output: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    assert output.ndim == 2 and target.ndim == 2
    output = output.detach().cpu().float()
    target = target.detach().cpu().float()
    predict = (output > 0.5).float()

    epsilon = 1e-3
    tp = ((predict == 1) & (target == 1)).sum(dim=0).float()
    fp = ((predict == 1) & (target == 0)).sum(dim=0).float()
    fn = ((predict == 0) & (target == 1)).sum(dim=0).float()
    tn = ((predict == 0) & (target == 0)).sum(dim=0).float()

    accuracy = (tp + tn) / (tp + fp + tn + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    specificity = tn / (tn + fp + epsilon)

    aurocs = []
    target_np = target.numpy()
    output_np = output.numpy()
    for y_true, y_score in zip(target_np, output_np):
        if np.unique(y_true).size < 2:
            continue
        aurocs.append(roc_auc_score(y_true, y_score))

    exact_match = (predict == target).all(dim=1).float().mean()
    return {
        "Accuracy_Macro": tensor_float(accuracy.mean()),
        "Precision_Macro": tensor_float(precision.mean()),
        "Recall_Macro": tensor_float(recall.mean()),
        "F1_Macro": tensor_float(f1.mean()),
        "Specificity_Macro": tensor_float(specificity.mean()),
        "Exact_Match": tensor_float(exact_match),
        "AUROC_Macro_from_binary_outputs": float(np.mean(aurocs)) if aurocs else float("nan"),
    }


def generate_reports(
    model,
    tokenizer,
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    model_device = first_model_device(model)
    eos_token_ids = generation_eos_ids(tokenizer)
    records = []

    for start in tqdm(range(0, len(examples), args.batch_size), desc="Generating reports"):
        batch_examples = examples[start : start + args.batch_size]
        prompts = [example["prompt"] for example in batch_examples]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
        )
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample,
            "eos_token_id": eos_token_ids,
            "pad_token_id": tokenizer.pad_token_id,
            "use_cache": True,
        }
        if args.do_sample:
            generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)

        prompt_length = encoded["input_ids"].shape[1]
        generated_ids = output_ids[:, prompt_length:]
        raw_generations = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        for example, raw_generation in zip(batch_examples, raw_generations):
            clean_report = clean_generated_report(raw_generation, tokenizer)
            record = dict(example)
            record.update(
                {
                    "raw_generation": raw_generation,
                    "clean_report": clean_report,
                }
            )
            records.append(record)
    return records


def add_parsed_labels(
    records: list[dict[str, Any]],
    names: list[str],
    diagnostic_agg_df: pd.DataFrame,
    code_to_superclass: dict[str, str],
    diagnostic_names: np.ndarray,
) -> None:
    superclass_names = [str(name) for name in diagnostic_names.tolist()]
    for record in records:
        record["parsed_prediction"] = resolve_diagnostic_labels(
            record["clean_report"],
            names=names,
            diagnostic_agg_df=diagnostic_agg_df,
            code_to_superclass=code_to_superclass,
            superclass_names=superclass_names,
        )
        record["parsed_reference"] = resolve_diagnostic_labels(
            record["reference_report"],
            names=names,
            diagnostic_agg_df=diagnostic_agg_df,
            code_to_superclass=code_to_superclass,
            superclass_names=superclass_names,
        )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.dump(payload, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def write_outputs(
    records: list[dict[str, Any]],
    metrics: dict[str, Any],
    output_prefix: Path,
) -> None:
    reports_json = output_prefix.with_name(output_prefix.name + "_reports.json")
    reports_yaml = output_prefix.with_name(output_prefix.name + "_reports.yaml")
    pred_json = output_prefix.with_name(output_prefix.name + "_pred_labels.json")
    ref_json = output_prefix.with_name(output_prefix.name + "_ref_labels.json")
    metrics_json = output_prefix.with_name(output_prefix.name + "_metrics.json")
    clean_txt = output_prefix.with_name(output_prefix.name + "_clean_reports.txt")

    write_json(reports_json, records)
    write_yaml(reports_yaml, records)
    write_json(pred_json, [record["parsed_prediction"] for record in records])
    write_json(ref_json, [record["parsed_reference"] for record in records])
    write_json(metrics_json, metrics)
    clean_txt.write_text(
        "\n\n".join(
            f"### ecg_id={record['ecg_id']}\n{record['clean_report']}" for record in records
        ),
        encoding="utf-8",
    )

    print(f"Wrote reports: {reports_json}")
    print(f"Wrote readable reports: {clean_txt}")
    print(f"Wrote metrics: {metrics_json}")


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    patch_phi3_dynamic_cache_compat()
    device = get_device()

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
    if torch.cuda.is_available():
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    combined_df, diagnostic_agg_df, code_to_superclass, names, diagnostic_names = (
        load_ptbxl_dataframe()
    )
    signal = load_signal()
    selected_df, selected_signal = select_split(
        combined_df,
        signal,
        split=args.split,
        max_samples=args.max_samples,
    )
    print(f"Selected split={args.split}, samples={len(selected_df)}")

    ecg_tokenizer = load_ecg_tokenizer(args.ecg_tokenizer_path, device=device)
    selected_df["tokens"] = tokenize_signals(
        selected_signal,
        ecg_tokenizer=ecg_tokenizer,
        batch_size=args.token_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    del ecg_tokenizer
    cleanup_cuda()

    selected_df["control_prompt"] = load_or_predict_control_prompts(
        selected_signal=selected_signal,
        expected_count=len(selected_df),
        args=args,
        device=device,
    )
    cleanup_cuda()

    text_tokenizer = load_text_tokenizer(args.merged_model_path)
    seed_everything(args.seed)
    examples = build_report_examples(selected_df, text_tokenizer)

    model = load_generation_model(args.merged_model_path)
    records = generate_reports(model, text_tokenizer, examples, args)
    add_parsed_labels(
        records,
        names=names,
        diagnostic_agg_df=diagnostic_agg_df,
        code_to_superclass=code_to_superclass,
        diagnostic_names=diagnostic_names,
    )

    label_onehot_predicted = compute_label_onehot(
        [record["parsed_prediction"] for record in records],
        diagnostic_names,
    ).float()
    label_onehot = compute_label_onehot(
        [record["parsed_reference"] for record in records],
        diagnostic_names,
    ).float()
    metrics = compute_macro_metrics_stable(label_onehot_predicted, label_onehot)
    metrics.update(
        {
            "split": args.split,
            "num_samples": len(records),
            "generation_mode": "autoregressive_model.generate",
            "max_new_tokens": args.max_new_tokens,
            "num_beams": args.num_beams,
            "do_sample": args.do_sample,
            "diagnostic_names": diagnostic_names.tolist(),
            "parse_empty_rate": float(
                np.mean([len(record["parsed_prediction"]) == 0 for record in records])
            )
            if records
            else float("nan"),
        }
    )

    write_outputs(records, metrics, args.output_prefix)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
