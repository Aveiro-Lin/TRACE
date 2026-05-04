# coding: utf-8

"""Autoregressive Ningbo structured-report inference for TRACE.

Ningbo does not provide public narrative reports, so this script evaluates the
trained TRACE language branch as structured diagnostic report generation. It uses
the same inference-time controls as training: refined diagnostic logits are
fused with the TRACE report-control head, rendered as control tokens, concatenated
with ECG tokens, and decoded with ``model.generate()``.
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
from torch.utils.data import ConcatDataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.cache_utils import DynamicCache

from utils.my_templates import Reports
from utils.trace_tokenizer import Tokenizer
from utils.report_controls_Ningbo import (
    DEFAULT_NINGBO_LABEL_NAMES_JSON,
    DEFAULT_NINGBO_ROOT,
    NingboReportControlDataset,
    infer_ningbo_label_dim,
    load_ningbo_label_names,
    parse_ningbo_label_vector,
)
from utils.report_controls import (
    DiagnosisProvider,
    build_control_token_maps,
    load_report_control_model,
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
DIAGNOSIS_THRESHOLD = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ECG_TOKENIZER_PATH = (
    PROJECT_ROOT
    / "ckpt/ningbo/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_Ningbo_st2_best.pth"
)
DEFAULT_MERGED_MODEL_PATH = PROJECT_ROOT / "ckpt/ningbo/report_merged_model"
DEFAULT_CONTROL_CHECKPOINT_PATH = PROJECT_ROOT / "ckpt/ningbo/report_control/best.pt"
DEFAULT_DIAGNOSIS_CHECKPOINT_PATH = PROJECT_ROOT / "ckpt/ningbo/downstream_refined/best.pt"
DEFAULT_OUTPUT_PREFIX = PROJECT_ROOT / "data/generated_Ningbo_st2"
DEFAULT_CONTROL_CACHE_PATH = PROJECT_ROOT / "data/control_prompts_Ningbo_st2_test.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate Ningbo TRACE structured reports.")
    parser.add_argument("--split", choices=["train", "valid", "test", "all"], default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--token_batch_size", type=int, default=64)
    parser.add_argument("--control_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=260)
    parser.add_argument("--max_prompt_length", type=int, default=3072)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ningbo_root", type=Path, default=Path(DEFAULT_NINGBO_ROOT))
    parser.add_argument("--label_names_json", type=Path, default=Path(DEFAULT_NINGBO_LABEL_NAMES_JSON))
    parser.add_argument("--merged_model_path", type=Path, default=DEFAULT_MERGED_MODEL_PATH)
    parser.add_argument("--control_checkpoint_path", type=Path, default=DEFAULT_CONTROL_CHECKPOINT_PATH)
    parser.add_argument("--diagnosis-checkpoint-path", type=Path, default=DEFAULT_DIAGNOSIS_CHECKPOINT_PATH)
    parser.add_argument("--ecg_tokenizer_path", type=Path, default=DEFAULT_ECG_TOKENIZER_PATH)
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
    if not hasattr(DynamicCache, "get_max_length"):
        DynamicCache.get_max_length = DynamicCache.get_max_cache_shape


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def require_path(path: str | Path, description: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def split_index_path(ningbo_root: Path, split: str) -> Path:
    name = "val" if split == "valid" else split
    return ningbo_root / f"{name}_index.csv"


def label_vector_to_dict(value, label_names: list[str]) -> str:
    vector = parse_ningbo_label_vector(value, expected_dim=len(label_names))
    return str({label: 100.0 for label, flag in zip(label_names, vector.tolist()) if flag})


def label_vector_to_names(value, label_names: list[str]) -> list[str]:
    vector = parse_ningbo_label_vector(value, expected_dim=len(label_names))
    return [label for label, flag in zip(label_names, vector.tolist()) if flag]


def build_split_frame(index_csv: Path, split: str, label_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(index_csv).reset_index(drop=True)
    df["diagnostic"] = df["LABEL"].apply(lambda value: label_vector_to_dict(value, label_names))
    df["form"] = "{}"
    df["rhythm"] = "{}"
    df["age"] = 0
    df["sex"] = "unknown"
    df["split"] = split
    df["true_labels"] = df["LABEL"].apply(lambda value: label_vector_to_names(value, label_names))
    return df


def build_signal_dataset(index_csv: Path, label_names: list[str], args: argparse.Namespace):
    return NingboReportControlDataset(
        index_csv,
        args.ningbo_root / "data",
        label_names=label_names,
        seq_length=SEQ_LENGTH,
        latent_ratio=LATENT_RATIO,
        source_rate=SAMPLING_RATE,
        target_rate=250,
        channels=CHANNELS,
        normalize=True,
        max_samples=args.max_samples,
    )


def load_selected_split(args: argparse.Namespace, label_names: list[str]) -> tuple[pd.DataFrame, Any]:
    if args.split == "all":
        frames = []
        datasets = []
        for split_name in ("train", "valid", "test"):
            index_csv = split_index_path(args.ningbo_root, split_name)
            frame = build_split_frame(index_csv, split_name, label_names)
            if args.max_samples is not None:
                frame = frame.iloc[: args.max_samples].copy()
            frames.append(frame)
            datasets.append(build_signal_dataset(index_csv, label_names, args))
        return pd.concat(frames, ignore_index=True), ConcatDataset(datasets)

    index_csv = split_index_path(args.ningbo_root, args.split)
    frame = build_split_frame(index_csv, args.split, label_names)
    if args.max_samples is not None:
        frame = frame.iloc[: args.max_samples].copy()
    return frame, build_signal_dataset(index_csv, label_names, args)


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
    tokenizer = Tokenizer(
        seq_length=SEQ_LENGTH,
        patch_size=PATCH_SIZE,
        latent_ratio=LATENT_RATIO,
        channels=CHANNELS,
        codebook_size=CODEBOOK_SIZE,
        vq_kwargs={"residual_levels": RESIDUAL_LEVELS},
    ).to(device)
    weights = torch.load(checkpoint_path, weights_only=False, map_location=device)
    tokenizer.load_state_dict(weights["model_state_dict"])
    tokenizer.eval()
    return tokenizer


def tokenize_dataset(
    dataset,
    *,
    ecg_tokenizer: Tokenizer,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> list[list[list[int]]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    tokens = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Tokenizing ECG"):
            signals = batch[0].to(device, non_blocking=True)
            batch_tokens = ecg_tokenizer.tokenize(signals).detach().cpu().tolist()
            tokens.extend(batch_tokens)
    return tokens


def load_or_predict_control_prompts(
    *,
    dataset,
    expected_count: int,
    label_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> list[str]:
    if args.control_cache_path.exists() and not args.refresh_control_cache:
        cached = json.loads(args.control_cache_path.read_text(encoding="utf-8"))
        if (
            cached.get("split") == args.split
            and cached.get("num_samples") == expected_count
            and cached.get("label_names") == label_names
        ):
            print(f"Loaded cached control prompts: {args.control_cache_path}")
            return list(cached["prompts"])

    control_model, control_vocab, control_checkpoint = load_report_control_model(
        args.control_checkpoint_path,
        device=device,
        tokenizer_checkpoint=str(args.ecg_tokenizer_path),
    )
    diagnostic_provider = DiagnosisProvider(
        checkpoint_path=args.diagnosis_checkpoint_path,
        tokenizer_checkpoint=str(args.ecg_tokenizer_path),
        device=device,
        class_names=label_names,
        seq_length=SEQ_LENGTH,
        patch_size=PATCH_SIZE,
        latent_ratio=LATENT_RATIO,
        channels=CHANNELS,
        codebook_size=CODEBOOK_SIZE,
        residual_levels=RESIDUAL_LEVELS,
        threshold=DIAGNOSIS_THRESHOLD,
        force_at_least_one=True,
    )
    control_threshold = float(control_checkpoint.get("args", {}).get("threshold", 0.5))
    loader = DataLoader(
        dataset,
        batch_size=args.control_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    prompts = predict_fused_control_prompts(
        diagnostic_provider=diagnostic_provider,
        control_model=control_model,
        loader=loader,
        vocab=control_vocab,
        device=device,
        control_threshold=control_threshold,
        use_amp=device.type == "cuda",
    )
    if len(prompts) != expected_count:
        raise RuntimeError(f"Control prompt count mismatch: got {len(prompts)}, expected {expected_count}")

    args.control_cache_path.parent.mkdir(parents=True, exist_ok=True)
    args.control_cache_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "num_samples": expected_count,
                "label_names": label_names,
                "prompts": prompts,
            },
            ensure_ascii=False,
            indent=2,
        ),
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
    return re.sub(r"<built-in method capitalize of str object at 0x[0-9a-fA-F]+>", "Provide", text)


def build_report_examples(
    selected_df: pd.DataFrame,
    text_tokenizer,
    control_vocab: dict[str, list[str]],
) -> list[dict[str, Any]]:
    token_maps = build_control_token_maps(control_vocab)["groups"]
    examples = []
    for row_idx, row in selected_df.reset_index(drop=True).iterrows():
        ecg_text = [f"<|ecg_{row.tokens[i][0] + 1}|>" for i in range(len(row.tokens))]
        ecg_input = "<|ecg_start|> " + " ".join(ecg_text) + " <|ecg_end|>"
        conditioned_ecg_input = f"{row.control_prompt} {ecg_input}"
        age = int(row.age)
        sex = str(row.sex)
        symptom_dict_list = [
            literal_dict(row["diagnostic"]),
            literal_dict(row["form"]),
            literal_dict(row["rhythm"]),
        ]
        template = Reports.get_template(age, sex, conditioned_ecg_input, symptom_dict_list)[0]
        user = {"content": sanitize_template_text(f"{template[0]}\n{template[1]}"), "role": "user"}
        assistant = {"content": sanitize_template_text(template[2]), "role": "assistant"}
        prompt_text = text_tokenizer.apply_chat_template([user], add_generation_prompt=True, tokenize=False)

        examples.append(
            {
                "index": int(row_idx),
                "split": str(row.split),
                "file_name": str(row.FILE_NAME),
                "relative_file_path": str(row.RELATIVE_FILE_PATH),
                "sample_rate": int(row.SAMPLE_RATE),
                "age": age,
                "sex": sex,
                "true_labels": list(row.true_labels),
                "control_labels": parse_control_prompt_labels(str(row.control_prompt), control_vocab, token_maps),
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
    text = text.replace(tokenizer.pad_token or "", "").strip()
    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]
    text = truncate_at_first_marker(text, ["<|end|>", "</s>", tokenizer.eos_token])
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def generate_reports(model, tokenizer, examples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    model_device = first_model_device(model)
    eos_token_ids = generation_eos_ids(tokenizer)
    records = []

    for start in tqdm(range(0, len(examples), args.batch_size), desc="Generating Ningbo reports"):
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
            record = dict(example)
            record.update(
                {
                    "raw_generation": raw_generation,
                    "clean_report": clean_generated_report(raw_generation, tokenizer),
                }
            )
            records.append(record)
    return records


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9]+", " ", str(text).upper())).strip()


def label_aliases(label: str) -> list[str]:
    aliases = [label]
    match = re.match(r"^(.*?)\s*\((.*?)\)\s*$", label)
    if match:
        base = match.group(1).strip()
        paren = match.group(2).strip()
        aliases.extend([base, paren])
        aliases.extend(part.strip() for part in re.split(r"[/,;]", paren) if part.strip())
    return sorted({alias for alias in aliases if alias}, key=len, reverse=True)


def diagnostic_section(text: str) -> str:
    match = re.search(
        r"(?:#+\s*)?Diagnostic\s+Recommendation\s*(.*?)(?:<\|end\|>|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return match.group(1).strip() if match else text


def diagnostic_item_lines(text: str) -> list[str]:
    section = diagnostic_section(text)
    lines = []
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*[-*]\s*", "", line).strip()
        line = re.sub(r"^\s*\d+[.)]\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines


def match_label_in_line(line: str, label: str) -> bool:
    normalized_line = normalize_for_match(line)
    if not normalized_line:
        return False
    match = re.match(r"^(.*?)\s*\((.*?)\)\s*$", label)
    if match:
        base = match.group(1).strip()
        paren = match.group(2).strip()
        normalized_base = normalize_for_match(base)
        normalized_full = normalize_for_match(label)
        if normalized_line in {normalized_base, normalized_full}:
            return True
        if normalized_base and normalized_line.startswith(f"{normalized_base} "):
            trailing = normalized_line[len(normalized_base) :].strip()
            paren_aliases = [normalize_for_match(part) for part in re.split(r"[/,;]", paren) if part.strip()]
            if trailing in paren_aliases or trailing == normalize_for_match(paren):
                return True
        for alias in [paren, *re.split(r"[/,;]", paren)]:
            alias = alias.strip()
            if not alias:
                continue
            if re.search(rf"\({re.escape(alias)}\)", line, flags=re.IGNORECASE):
                return True
            if normalized_line == normalize_for_match(alias):
                return True
        return False

    return normalized_line == normalize_for_match(label)


def parse_report_labels(text: str, label_names: list[str]) -> list[str]:
    lines = diagnostic_item_lines(text)
    labels = []
    for label in label_names:
        if any(match_label_in_line(line, label) for line in lines):
            labels.append(label)
    if labels:
        return labels

    normalized_text = normalize_for_match(diagnostic_section(text))
    for label in label_names:
        normalized_full = normalize_for_match(label)
        if normalized_full and re.search(rf"(?<![A-Z0-9]){re.escape(normalized_full)}(?![A-Z0-9])", normalized_text):
            labels.append(label)
            continue
        for alias in label_aliases(label):
            normalized_alias = normalize_for_match(alias)
            if normalized_alias and normalized_text == normalized_alias:
                labels.append(label)
                break
    return labels


def parse_control_prompt_labels(
    prompt: str,
    control_vocab: dict[str, list[str]],
    token_maps: dict[str, object] | None = None,
) -> list[str]:
    if token_maps is None:
        token_maps = build_control_token_maps(control_vocab)["groups"]
    diagnostic_tokens = token_maps["diagnostic"]["label_to_token"]
    return [label for label in control_vocab.get("diagnostic", []) if diagnostic_tokens[label] in prompt]


def labels_to_onehot(label_lists: list[list[str]], label_names: list[str]) -> torch.Tensor:
    label_to_index = {label: idx for idx, label in enumerate(label_names)}
    onehot = torch.zeros((len(label_lists), len(label_names)), dtype=torch.float32)
    for row_idx, labels in enumerate(label_lists):
        for label in labels:
            if label in label_to_index:
                onehot[row_idx, label_to_index[label]] = 1.0
    return onehot


def tensor_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def compute_macro_metrics(output: torch.Tensor, target: torch.Tensor, label_names: list[str]) -> dict[str, Any]:
    output = output.detach().cpu().float()
    target = target.detach().cpu().float()
    predict = (output > 0.5).float()
    eps = 1e-6
    tp = ((predict == 1) & (target == 1)).sum(dim=0).float()
    fp = ((predict == 1) & (target == 0)).sum(dim=0).float()
    fn = ((predict == 0) & (target == 1)).sum(dim=0).float()
    tn = ((predict == 0) & (target == 0)).sum(dim=0).float()

    accuracy = (tp + tn) / (tp + fp + tn + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    specificity = tn / (tn + fp + eps)

    target_np = target.numpy()
    output_np = output.numpy()
    aurocs = []
    for class_idx in range(target_np.shape[1]):
        if np.unique(target_np[:, class_idx]).size < 2:
            continue
        aurocs.append(roc_auc_score(target_np[:, class_idx], output_np[:, class_idx]))

    per_class = []
    for idx, label in enumerate(label_names):
        per_class.append(
            {
                "label": label,
                "support": int(target[:, idx].sum().item()),
                "predicted_positive": int(predict[:, idx].sum().item()),
                "precision": tensor_float(precision[idx]),
                "recall": tensor_float(recall[idx]),
                "f1": tensor_float(f1[idx]),
                "accuracy": tensor_float(accuracy[idx]),
            }
        )

    return {
        "Accuracy_Macro": tensor_float(accuracy.mean()),
        "Precision_Macro": tensor_float(precision.mean()),
        "Recall_Macro": tensor_float(recall.mean()),
        "F1_Macro": tensor_float(f1.mean()),
        "Specificity_Macro": tensor_float(specificity.mean()),
        "Exact_Match": tensor_float((predict == target).all(dim=1).float().mean()),
        "AUROC_Macro_from_binary_outputs": float(np.mean(aurocs)) if aurocs else float("nan"),
        "per_class": per_class,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(payload, allow_unicode=True, default_flow_style=False), encoding="utf-8")


def write_outputs(records: list[dict[str, Any]], metrics: dict[str, Any], output_prefix: Path) -> None:
    reports_json = output_prefix.with_name(output_prefix.name + "_reports.json")
    reports_yaml = output_prefix.with_name(output_prefix.name + "_reports.yaml")
    pred_json = output_prefix.with_name(output_prefix.name + "_pred_labels.json")
    ref_json = output_prefix.with_name(output_prefix.name + "_ref_labels.json")
    metrics_json = output_prefix.with_name(output_prefix.name + "_metrics.json")
    clean_txt = output_prefix.with_name(output_prefix.name + "_clean_reports.txt")

    write_json(reports_json, records)
    write_yaml(reports_yaml, records)
    write_json(pred_json, [record["parsed_prediction"] for record in records])
    write_json(ref_json, [record["true_labels"] for record in records])
    write_json(metrics_json, metrics)
    clean_txt.write_text(
        "\n\n".join(
            f"### index={record['index']} file={record['file_name']}\n{record['clean_report']}"
            for record in records
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

    for path, description in [
        (args.merged_model_path, "Ningbo merged Phi-3 model"),
        (args.control_checkpoint_path, "Ningbo report-control checkpoint"),
        (args.diagnosis_checkpoint_path, "Ningbo diagnosis checkpoint"),
        (args.ecg_tokenizer_path, "Ningbo ECG tokenizer checkpoint"),
        (args.ningbo_root / "data", "Ningbo waveform directory"),
        (args.label_names_json, "Ningbo label names JSON"),
    ]:
        require_path(path, description)

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
    if torch.cuda.is_available():
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    label_dim = infer_ningbo_label_dim(split_index_path(args.ningbo_root, "train"))
    label_names = load_ningbo_label_names(args.label_names_json, label_dim)
    selected_df, signal_dataset = load_selected_split(args, label_names)
    print(json.dumps({"split": args.split, "num_samples": len(selected_df), "num_labels": len(label_names)}, ensure_ascii=False))

    ecg_tokenizer = load_ecg_tokenizer(args.ecg_tokenizer_path, device=device)
    selected_df["tokens"] = tokenize_dataset(
        signal_dataset,
        ecg_tokenizer=ecg_tokenizer,
        batch_size=args.token_batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    del ecg_tokenizer
    cleanup_cuda()

    control_model_for_vocab, control_vocab, _control_checkpoint = load_report_control_model(
        args.control_checkpoint_path,
        device=device,
        tokenizer_checkpoint=str(args.ecg_tokenizer_path),
    )
    del control_model_for_vocab
    cleanup_cuda()

    selected_df["control_prompt"] = load_or_predict_control_prompts(
        dataset=signal_dataset,
        expected_count=len(selected_df),
        label_names=label_names,
        args=args,
        device=device,
    )
    cleanup_cuda()

    text_tokenizer = load_text_tokenizer(args.merged_model_path)
    seed_everything(args.seed)
    examples = build_report_examples(selected_df, text_tokenizer, control_vocab)

    model = load_generation_model(args.merged_model_path)
    records = generate_reports(model, text_tokenizer, examples, args)
    for record in records:
        record["parsed_prediction"] = parse_report_labels(record["clean_report"], label_names)
        record["parsed_reference"] = parse_report_labels(record["reference_report"], label_names)

    generated_onehot = labels_to_onehot([record["parsed_prediction"] for record in records], label_names)
    true_onehot = labels_to_onehot([record["true_labels"] for record in records], label_names)
    control_onehot = labels_to_onehot([record["control_labels"] for record in records], label_names)

    report_metrics = compute_macro_metrics(generated_onehot, true_onehot, label_names)
    control_metrics = compute_macro_metrics(control_onehot, true_onehot, label_names)
    metrics = {
        "split": args.split,
        "num_samples": len(records),
        "generation_mode": "autoregressive_model.generate",
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "diagnostic_names": label_names,
        "parse_empty_rate": float(np.mean([len(record["parsed_prediction"]) == 0 for record in records]))
        if records
        else float("nan"),
        "report_vs_true": report_metrics,
        "control_prompt_vs_true": control_metrics,
    }

    write_outputs(records, metrics, args.output_prefix)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
