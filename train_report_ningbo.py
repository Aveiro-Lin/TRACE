# coding: utf-8

import ast
import gc
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, TaskType
from safetensors.torch import load_file
from torch.utils.data import ConcatDataset, DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed
from trl import SFTTrainer

from utils.my_templates import Choices, Predict, Reports, Sentences
from utils.trace_tokenizer import Tokenizer
from utils.report_controls_Ningbo import (
    DEFAULT_NINGBO_LABEL_NAMES_JSON,
    DEFAULT_NINGBO_ROOT,
    NingboReportControlDataset,
    build_ningbo_control_vocab,
    infer_ningbo_label_dim,
    load_ningbo_label_names,
    parse_ningbo_label_vector,
)
from utils.report_controls import (
    DiagnosisProvider,
    build_control_token_maps,
    load_report_control_model,
    predict_fused_control_prompts,
    render_control_prompt_from_row,
    save_control_vocab,
)

set_seed(42)


print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
if torch.cuda.is_available():
    print(f"Current device index: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


sampling_rate = 500
batch_size = 32
seq_length = 2250
patch_size = 75
latent_ratio = 0.1
channels = 12
codebook_size = 256
residual_levels = 2
num_train_epochs = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = os.environ.get("TRACE_BASE_LM_PATH", "path/to/Phi-3-mini-4k-instruct")
num = 10
trainer_path = f"ckpt/ningbo/report_lora_adapter"
merged_model_path = "ckpt/ningbo/report_merged_model"

ecg_tokenizer_path = "ckpt/ningbo/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_Ningbo_st2_best.pth"
control_vocab_path = "ckpt/ningbo/report_control/report_control_vocab_Ningbo.json"
control_checkpoint_path = "ckpt/ningbo/report_control/best.pt"
diagnosis_checkpoint_path = "ckpt/ningbo/downstream_refined/best.pt"
diagnosis_threshold = 0.5
train_pred_control_ratio = 0.35

ningbo_root = Path(DEFAULT_NINGBO_ROOT)
train_index = ningbo_root / "train_index.csv"
valid_index = ningbo_root / "val_index.csv"
test_index = ningbo_root / "test_index.csv"
label_names_json = DEFAULT_NINGBO_LABEL_NAMES_JSON


def require_path(path: str | Path, description: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def label_vector_to_dict(value, label_names: list[str]) -> str:
    vector = parse_ningbo_label_vector(value, expected_dim=len(label_names))
    return str({label: 100.0 for label, flag in zip(label_names, vector.tolist()) if flag})


def build_split_frame(index_csv: Path, split: str, label_names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(index_csv).reset_index(drop=True)
    df["diagnostic"] = df["LABEL"].apply(lambda value: label_vector_to_dict(value, label_names))
    df["form"] = "{}"
    df["rhythm"] = "{}"
    df["age"] = 0
    df["sex"] = "unknown"
    df["split"] = split
    return df


def build_ningbo_signal_dataset(index_csv: Path, label_names: list[str]) -> NingboReportControlDataset:
    return NingboReportControlDataset(
        index_csv,
        ningbo_root / "data",
        label_names=label_names,
        seq_length=seq_length,
        latent_ratio=latent_ratio,
        source_rate=sampling_rate,
        target_rate=250,
        channels=channels,
        normalize=True,
    )


for path, description in [
    (model_path, "Phi-3 model path"),
    (ecg_tokenizer_path, "Ningbo ECG tokenizer checkpoint"),
    (control_checkpoint_path, "Ningbo report-control checkpoint"),
    (diagnosis_checkpoint_path, "Ningbo diagnosis checkpoint"),
    (train_index, "Ningbo train index"),
    (valid_index, "Ningbo validation index"),
    (test_index, "Ningbo test index"),
    (ningbo_root / "data", "Ningbo waveform directory"),
]:
    require_path(path, description)

label_dim = infer_ningbo_label_dim(train_index)
label_names = load_ningbo_label_names(label_names_json, label_dim)
control_vocab = build_ningbo_control_vocab(label_names)
save_control_vocab(control_vocab_path, control_vocab)

combined_df_train = build_split_frame(train_index, "train", label_names)
combined_df_valid = build_split_frame(valid_index, "valid", label_names)
combined_df_test = build_split_frame(test_index, "test", label_names)
combined_df = pd.concat([combined_df_train, combined_df_valid, combined_df_test], ignore_index=True)

train_dataset_signal = build_ningbo_signal_dataset(train_index, label_names)
valid_dataset_signal = build_ningbo_signal_dataset(valid_index, label_names)
test_dataset_signal = build_ningbo_signal_dataset(test_index, label_names)
all_signal_dataset = ConcatDataset([train_dataset_signal, valid_dataset_signal, test_dataset_signal])

print(
    json.dumps(
        {
            "dataset": "Ningbo",
            "num_train": len(train_dataset_signal),
            "num_valid": len(valid_dataset_signal),
            "num_test": len(test_dataset_signal),
            "num_labels": len(label_names),
            "note": "Ningbo has diagnostic labels only; form/rhythm controls are empty and reports are structured diagnostic reports.",
        },
        ensure_ascii=False,
    ),
    flush=True,
)


vq_kwargs = {"residual_levels": residual_levels}
ecg_tokenizer = Tokenizer(
    seq_length=seq_length,
    patch_size=patch_size,
    latent_ratio=latent_ratio,
    channels=channels,
    codebook_size=codebook_size,
    vq_kwargs=vq_kwargs,
).to(device)
ecg_weights = torch.load(ecg_tokenizer_path, weights_only=False)
ecg_tokenizer.load_state_dict(ecg_weights["model_state_dict"])
ecg_tokenizer.eval()

os.makedirs("ckpt/ningbo/tokenizer", exist_ok=True)
torch.save({"model_state_dict": ecg_tokenizer.state_dict()}, "ckpt/ningbo/tokenizer/tok_2_Ningbo_st2.pth")


config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
text_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
text_tokenizer.pad_token = text_tokenizer.unk_token
text_tokenizer.pad_token_id = text_tokenizer.convert_tokens_to_ids(text_tokenizer.pad_token)
text_tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    attn_implementation="eager",
).to(device)

weight1 = load_file(f"{model_path}/model-00001-of-00002.safetensors")
weight2 = load_file(f"{model_path}/model-00002-of-00002.safetensors")
model.load_state_dict({**weight1, **weight2}, strict=False)

args = TrainingArguments(
    output_dir=trainer_path,
    logging_dir=f"runs/{trainer_path}",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    do_eval=True,
    optim="adamw_torch",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    per_device_eval_batch_size=4,
    log_level="warning",
    logging_steps=100,
    learning_rate=1e-4,
    fp16=True,
    bf16=False,
    num_train_epochs=num_train_epochs,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    report_to="tensorboard",
    seed=42,
)

peft_config = LoraConfig(
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "q_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
)


signal_loader = DataLoader(all_signal_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
tokens = []
with torch.no_grad():
    for batch in signal_loader:
        x = batch[0].to(device, non_blocking=True)
        ecg_tokens = ecg_tokenizer.tokenize(x).cpu().detach().tolist()
        tokens.extend(ecg_tokens)

if len(tokens) != len(combined_df):
    raise RuntimeError(f"Token count mismatch: got {len(tokens)}, expected {len(combined_df)}")
combined_df["tokens"] = tokens

control_token_spec = build_control_token_maps(control_vocab)
combined_df["gt_control_prompt"] = combined_df.apply(
    lambda row: render_control_prompt_from_row(row, control_vocab, control_token_spec["groups"]),
    axis=1,
)

control_model, loaded_control_vocab, control_checkpoint = load_report_control_model(
    control_checkpoint_path,
    device=device,
    tokenizer_checkpoint=ecg_tokenizer_path,
)
if loaded_control_vocab != control_vocab:
    print("Loaded control vocab differs from freshly built vocab; using checkpoint vocab for prediction alignment.")
    control_vocab = loaded_control_vocab
    control_token_spec = build_control_token_maps(control_vocab)
    combined_df["gt_control_prompt"] = combined_df.apply(
        lambda row: render_control_prompt_from_row(row, control_vocab, control_token_spec["groups"]),
        axis=1,
    )

diagnostic_provider = DiagnosisProvider(
    checkpoint_path=diagnosis_checkpoint_path,
    tokenizer_checkpoint=ecg_tokenizer_path,
    device=device,
    class_names=label_names,
    seq_length=seq_length,
    patch_size=patch_size,
    latent_ratio=latent_ratio,
    channels=channels,
    codebook_size=codebook_size,
    residual_levels=residual_levels,
    threshold=diagnosis_threshold,
    force_at_least_one=True,
)
control_threshold = float(control_checkpoint.get("args", {}).get("threshold", 0.5))
combined_df["pred_control_prompt"] = predict_fused_control_prompts(
    diagnostic_provider=diagnostic_provider,
    control_model=control_model,
    loader=signal_loader,
    vocab=control_vocab,
    device=device,
    control_threshold=control_threshold,
    use_amp=device.type == "cuda",
)
del control_model
del diagnostic_provider

train_mask = combined_df["split"] == "train"
valid_mask = combined_df["split"] == "valid"
test_mask = combined_df["split"] == "test"
train_rng = np.random.default_rng(42)
train_use_pred = train_rng.random(int(train_mask.sum())) < train_pred_control_ratio
train_indices = combined_df.index[train_mask]
combined_df["control_prompt"] = combined_df["gt_control_prompt"]
for index, use_pred in zip(train_indices, train_use_pred):
    if use_pred:
        combined_df.at[index, "control_prompt"] = combined_df.at[index, "pred_control_prompt"]
combined_df.loc[valid_mask | test_mask, "control_prompt"] = combined_df.loc[
    valid_mask | test_mask, "pred_control_prompt"
]


new_tokens = (
    ["<|ecg_start|>", "<|ecg_end|>", "<|report_start|>", "<|report_end|>", "<|pred_start|>", "<|pred_end|>"]
    + [f"<|ecg_{i + 1}|>" for i in range(codebook_size)]
    + control_token_spec["special_tokens"]
)
text_tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(text_tokenizer))


def message_form(template):
    user = {"content": f"{template[0]}\n{template[1]}", "role": "user"}
    assistant = {"content": template[2], "role": "assistant"}
    return {"text": text_tokenizer.apply_chat_template([user, assistant], add_generation_prompt=False, tokenize=False)}


symptoms = ["diagnostic", "form", "rhythm"]


def create_message_column(row):
    ecg_text = [f"<|ecg_{row.tokens[i][0] + 1}|>" for i in range(len(row.tokens))]
    ecg_input = "<|ecg_start|> " + " ".join(ecg_text) + " <|ecg_end|>"
    conditioned_ecg_input = row.control_prompt + " " + ecg_input
    age = int(row.age)
    sex = str(row.sex)
    symptom_dict_list = [ast.literal_eval(row[symptoms[idx]]) for idx in range(len(symptoms))]
    sentence_items = Sentences.get_template(age, sex, conditioned_ecg_input, symptom_dict_list)
    choice_items = Choices.get_template(age, sex, conditioned_ecg_input, symptom_dict_list)
    report_items = Reports.get_template(age, sex, conditioned_ecg_input, symptom_dict_list)
    predict_items = Predict.get_template(age, sex, conditioned_ecg_input)
    return [message_form(item) for item in sentence_items + choice_items + report_items + predict_items]


dataset_chatml = combined_df.apply(create_message_column, axis=1)
train_list = dataset_chatml[train_mask]
valid_list = dataset_chatml[valid_mask]
test_list = dataset_chatml[test_mask]

train_dataset = Dataset.from_list([item for sublist in train_list for item in sublist]).shuffle(seed=42)
valid_dataset = Dataset.from_list([item for sublist in valid_list for item in sublist])
test_dataset = Dataset.from_list([item for sublist in test_list for item in sublist])
print(
    json.dumps(
        {
            "train_messages": len(train_dataset),
            "valid_messages": len(valid_dataset),
            "test_messages": len(test_dataset),
        },
        ensure_ascii=False,
    ),
    flush=True,
)


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=peft_config,
    processing_class=text_tokenizer,
    args=args,
)

trainer.train()
trainer.save_model()

del model.__dict__
del model
del trainer
gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(torch.device("cuda"))
    print("CUDA device reset")
else:
    print("CUDA unavailable")

gc.collect()

new_model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
merged_model = new_model.merge_and_unload()
merged_model.save_pretrained(merged_model_path, trust_remote_code=True, safe_serialization=True)
text_tokenizer.save_pretrained(merged_model_path)
