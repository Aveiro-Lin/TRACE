# coding: utf-8

# In[17]:


import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import numpy as np
import random
import math
import ast

import torch
from torch.utils.data import DataLoader, TensorDataset
import utils.my_ecg_process as ecg
from utils.trace_tokenizer import Tokenizer
from utils.my_templates import Sentences, Choices, Reports, Predict
from utils.report_controls import (
    add_ptbxl_diagnostic_superclass_column,
    build_control_token_maps,
    build_ptbxl_v15_vocab,
    load_diagnostic_provider,
    load_report_control_model,
    predict_fused_control_prompts,
    render_control_prompt_from_row,
    save_control_vocab,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, TrainingArguments, set_seed
from safetensors.torch import load_file
from peft import LoraConfig, TaskType
from trl import SFTTrainer
from datasets import Dataset

set_seed(42)


# In[18]:


print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
print(f"PyTorch sees {torch.cuda.device_count()} GPU(s)")
if torch.cuda.is_available():
    print(f"Current device index: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


# In[19]:


sampling_rate = 500
batch_size = 32
# batch_size = 1
# seq_length = 500
# seq_length = 1500
seq_length = 2250
# patch_size = 25
patch_size = 75
# latent_ratio = 0.5
latent_ratio = 0.1
channels = 12
codebook_size = 256
residual_levels = 2
num_train_epochs = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dir = f"level_{residual_levels}_code_{codebook_size}_len_{seq_length}_ratio_{latent_ratio}"
ecg_tokenizer_path = f"ckpt/ptbxl/tokenizer/ecg_tokenizer_{dir}_st2.pth"
model_path = os.environ.get('TRACE_BASE_LM_PATH', 'path/to/Phi-3-mini-4k-instruct')
num = 10
trainer_path = f"ckpt/ptbxl/report_lora_adapter"
merged_model_path = "ckpt/ptbxl/report_merged_model"
control_vocab_path = "ckpt/ptbxl/report_control/report_control_vocab_ptbxl.json"
control_checkpoint_path = "ckpt/ptbxl/report_control/best.pt"
diagnosis_checkpoint_path = "ckpt/ptbxl/downstream_refined/best.pt"
diagnosis_threshold = 0.5
control_column_map = {'diagnostic': 'diagnostic_superclass', 'form': 'form', 'rhythm': 'rhythm'}
min_form_count = 300
min_rhythm_count = 100
train_pred_control_ratio = 0.35
valid_fold = 9
test_fold = 10


# In[20]:


vq_kwargs = {'residual_levels': residual_levels}
ecg_tokenizer = Tokenizer(
    seq_length=seq_length,
    patch_size=patch_size,
    latent_ratio=latent_ratio,
    channels=channels,
    codebook_size=codebook_size,
    vq_kwargs=vq_kwargs
).to(device)
ecg_weights = torch.load(ecg_tokenizer_path, weights_only=False)
ecg_tokenizer.load_state_dict(ecg_weights['model_state_dict'])


# In[5]:


os.makedirs('ckpt/ptbxl/tokenizer', exist_ok=True)
tok = {
    'model_state_dict': ecg_tokenizer.state_dict(),
}
torch.save(tok, "ckpt/ptbxl/tokenizer/tok_2_st2.pth")


# In[6]:


config = AutoConfig.from_pretrained(
    model_path,
    # trust_remote_code=False,
    trust_remote_code=True,
)

text_tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    # trust_remote_code=False,
    trust_remote_code=True,
)

text_tokenizer.pad_token = text_tokenizer.unk_token
text_tokenizer.pad_token_id = text_tokenizer.convert_tokens_to_ids(text_tokenizer.pad_token)
text_tokenizer.padding_side = 'left'


# In[7]:


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    config=config,
    # trust_remote_code=False,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    # attn_implementation='sdpa',
    attn_implementation='eager'
    # device_map="auto"
).to(device)

weight1 = load_file(f"{model_path}/model-00001-of-00002.safetensors")
weight2 = load_file(f"{model_path}/model-00002-of-00002.safetensors")

state_dict = {**weight1, **weight2}
model.load_state_dict(state_dict, strict=False)


# In[8]:


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
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"],
)


# In[9]:


# # total_length = int(seq_length * (1 + latent_ratio))
# Y = pd.read_csv('data/ptbxl_database.csv', index_col='ecg_id')
# X = ecg.load_raw_data(Y, 'data/ptbxl/', sampling_rate, seq_length=None)
# np.save("data/full_data.npy", X)


# In[10]:


X = np.load("data/full_data.npy")
# X = np.load('data/records500.npy')
total_length = int(seq_length * (1 + latent_ratio))
start = int((X.shape[2] - total_length)/2)
signal = X[:, :, start:start+seq_length]
predict = X[:, :, start+seq_length:start+total_length]


# In[11]:


df_dataset = TensorDataset(torch.tensor(signal, dtype=torch.float32))
df_loader = DataLoader(df_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# In[12]:


tokens = []
new = []
with torch.no_grad(): 
    for batch in df_loader:
        x = batch[0].to(device)
        ecg_tokens = ecg_tokenizer.tokenize(x).cpu().detach().tolist()
        tokens.extend(ecg_tokens)        


# In[ ]:


def compute_label_aggregations(df,folder):
    
    ctype='rhythm'
    
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))
    symptoms = ["diagnostic", "form", "rhythm"]

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]
    agg_df = {symptom: aggregation_df[aggregation_df[symptom] == 1.0] for symptom in symptoms}

    def aggregate_rhythm(y_dic):
        # tmp = []
        tmp = {}
        for key in y_dic.keys():
            if key in rhythm_agg_df.index:
                c = key
                if str(c) != 'nan':
                    # tmp.append(c)
                    tmp.update([(c, y_dic[c])])
        # return list(set(tmp))
        return str(tmp)
    
    def aggregate(symptom='rhythm'):
        def _aggregate(y_dic):
            # tmp = []
            tmp = {}
            for key in y_dic.keys():
                if key in agg_df[symptom].index:
                    c = key
                    if str(c) != 'nan':
                        # tmp.append(c)
                        tmp.update([(c, y_dic[c])])
            # return list(set(tmp))
            return str(tmp)
        return _aggregate

    # df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
    # df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    for symptom in symptoms:
        df[symptom] = df.scp_codes.apply(aggregate(symptom=symptom))

    return df

def handle_sex(sex):
    if isinstance(sex, str):
        return sex
    elif isinstance(sex, int):
        return 'male' if sex == 1 else 'female'
    else:
        raise NotImplementedError

# combined_df = pd.read_parquet("data/full_data.parquet")

combined_df = pd.read_csv('data/ptbxl_database.csv', index_col='ecg_id')
combined_df.scp_codes = combined_df.scp_codes.apply(lambda x: ast.literal_eval(x))
combined_df = compute_label_aggregations(combined_df, 'data/')
combined_df = add_ptbxl_diagnostic_superclass_column(combined_df, 'data/scp_statements.csv')
combined_df.sex = combined_df.sex.apply(handle_sex)


# In[ ]:


X = np.load("data/full_data.npy")
# X = np.load('data/records500.npy')
total_length = int(seq_length * (1 + latent_ratio))
start = int((X.shape[2] - total_length)/2)
signal = X[:, :, start:start+seq_length]
predict = X[:, :, start+seq_length:start+total_length]


# In[ ]:


df_dataset = TensorDataset(torch.tensor(signal, dtype=torch.float32))
df_loader = DataLoader(df_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)


# In[ ]:


tokens = []
new = []
with torch.no_grad(): 
    for batch in df_loader:
        x = batch[0].to(device)
        ecg_tokens = ecg_tokenizer.tokenize(x).cpu().detach().tolist()
        tokens.extend(ecg_tokens)
        
combined_df["tokens"] = tokens
train_mask = (combined_df.strat_fold != valid_fold) & (combined_df.strat_fold != test_fold)
valid_mask = combined_df.strat_fold == valid_fold
test_mask = combined_df.strat_fold == test_fold

control_vocab = build_ptbxl_v15_vocab(
    combined_df[train_mask],
    min_form_count=min_form_count,
    min_rhythm_count=min_rhythm_count,
)
save_control_vocab(control_vocab_path, control_vocab)
control_token_spec = build_control_token_maps(control_vocab)
combined_df["gt_control_prompt"] = combined_df.apply(
    lambda row: render_control_prompt_from_row(
        row,
        control_vocab,
        control_token_spec['groups'],
        column_map=control_column_map,
    ),
    axis=1,
)

if os.path.exists(control_checkpoint_path):
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
            lambda row: render_control_prompt_from_row(
                row,
                control_vocab,
                control_token_spec['groups'],
                column_map=control_column_map,
            ),
            axis=1,
        )
    if not os.path.exists(diagnosis_checkpoint_path):
        raise FileNotFoundError(
            f"TRACE requires the diagnosis checkpoint: {diagnosis_checkpoint_path}"
        )
    diagnostic_provider = load_diagnostic_provider(
        diagnosis_checkpoint_path,
        tokenizer_checkpoint=ecg_tokenizer_path,
        device=device,
        dataset='ptbxl',
        seq_length=seq_length,
        patch_size=patch_size,
        latent_ratio=latent_ratio,
        channels=channels,
        codebook_size=codebook_size,
        residual_levels=residual_levels,
        threshold=diagnosis_threshold,
        force_at_least_one=True,
    )
    control_loader = DataLoader(
        df_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    control_threshold = float(control_checkpoint.get('args', {}).get('threshold', 0.5))
    combined_df["pred_control_prompt"] = predict_fused_control_prompts(
        diagnostic_provider=diagnostic_provider,
        control_model=control_model,
        loader=control_loader,
        vocab=control_vocab,
        device=device,
        control_threshold=control_threshold,
        use_amp=device.type == 'cuda',
    )
    del control_model
    del diagnostic_provider
else:
    raise FileNotFoundError(
        f"TRACE requires the report control checkpoint for form/rhythm prediction: {control_checkpoint_path}"
    )

train_rng = np.random.default_rng(42)
train_use_pred = train_rng.random(train_mask.sum()) < train_pred_control_ratio
train_indices = combined_df.index[train_mask]
combined_df["control_prompt"] = combined_df["gt_control_prompt"]
for index, use_pred in zip(train_indices, train_use_pred):
    if use_pred:
        combined_df.at[index, "control_prompt"] = combined_df.at[index, "pred_control_prompt"]
combined_df.loc[valid_mask | test_mask, "control_prompt"] = combined_df.loc[valid_mask | test_mask, "pred_control_prompt"]


# In[ ]:


new_tokens = (
    ["<|ecg_start|>", "<|ecg_end|>", "<|report_start|>", "<|report_end|>", "<|pred_start|>", "<|pred_end|>"]
    + [f"<|ecg_{i+1}|>" for i in range(codebook_size)]
    + control_token_spec['special_tokens']
)
text_tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(text_tokenizer))


# In[ ]:


def message_form(template):
    user = {
        "content": f"{template[0]}\n{template[1]}",
        "role": "user"
    }

    assistant = {
        "content": template[2],
        "role": "assistant"
    }

    msg = {"text": text_tokenizer.apply_chat_template([user, assistant], add_generation_prompt=False, tokenize=False)}
    return msg


# In[ ]:


symptoms = ["diagnostic", "form", "rhythm"]
# symptoms = ["scp_codes"]

def create_message_column(row):
    messages = []
    ecg_text = [f"<|ecg_{row.tokens[i][0]+1}|>" for i in range(len(row.tokens))]
    ecg_input = "<|ecg_start|> " + " ".join(ecg_text) + " <|ecg_end|>"
    control_input = row.control_prompt
    conditioned_ecg_input = control_input + " " + ecg_input
    age = int(row.age) if not np.isnan(row.age) else 0
    sex = row.sex

    symptom_dict_list = [
        ast.literal_eval(row["diagnostic_superclass"]),
        ast.literal_eval(row["form"]),
        ast.literal_eval(row["rhythm"]),
    ]
    S = Sentences.get_template(age, sex, conditioned_ecg_input, symptom_dict_list)
    C = Choices.get_template(age, sex, conditioned_ecg_input, symptom_dict_list)
    R = Reports.get_template(age, sex, conditioned_ecg_input, symptom_dict_list)
    P = Predict.get_template(age, sex, conditioned_ecg_input)
    lists = S + C + R + P
    messages = [message_form(l) for l in lists]
    return messages


# In[ ]:


dataset_chatml = combined_df.apply(create_message_column, axis=1)


# In[ ]:


train_list = dataset_chatml[(combined_df.strat_fold != valid_fold) & (combined_df.strat_fold != test_fold)]
valid_list = dataset_chatml[combined_df.strat_fold == valid_fold]
test_list = dataset_chatml[combined_df.strat_fold == test_fold]

train_dataset = Dataset.from_list([item for sublist in train_list for item in sublist]).shuffle(seed=42)
valid_dataset = Dataset.from_list([item for sublist in valid_list for item in sublist])
test_dataset = Dataset.from_list([item for sublist in test_list for item in sublist])


# In[ ]:


trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
        processing_class=text_tokenizer,
        args=args,
)


# In[ ]:


trainer.train()
trainer.save_model()


# In[ ]:


del model.__dict__
del model
del trainer

import gc
gc.collect()
gc.collect()
torch.cuda.empty_cache()
gc.collect()

if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    device = torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    print("CUDA 设备已重置")
else:
    print("CUDA 不可用")
    
gc.collect()


# In[ ]:


from peft import AutoPeftModelForCausalLM

new_model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    trust_remote_code=True
)
merged_model = new_model.merge_and_unload()
merged_model.save_pretrained(merged_model_path, trust_remote_code=True, safe_serialization=True)
text_tokenizer.save_pretrained(merged_model_path)
