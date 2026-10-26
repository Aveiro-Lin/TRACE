#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TRACE_PYTHON:-python3}"
gpu_id="${TRACE_GPU:-0}"

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$gpu_id"

exec "$python_bin" finetune_anzhen_diagnosis.py \
  --data-root "${ANZHEN_DATA_ROOT:-data/anzhen_external}" \
  --source-checkpoint "${PTBXL_DIAGNOSIS_CHECKPOINT:-ckpt/ptbxl/downstream_refined/best.pt}" \
  --tokenizer-checkpoint "${PTBXL_TOKENIZER_CHECKPOINT:-ckpt/ptbxl/tokenizer/ecg_tokenizer_level_2_code_256_len_2250_ratio_0.1_st2.pth}" \
  --split-protocol 50_50 \
  --seed "${SPLIT_SEED:-42}" \
  --split-trials "${SPLIT_TRIALS:-512}" \
  --output-dir "${ANZHEN_DIAGNOSIS_OUTPUT_DIR:-results/anzhen_diagnosis/split_50_50_seed42}" \
  --epochs "${EPOCHS:-30}" \
  --patience "${PATIENCE:-7}" \
  --min-epochs "${MIN_EPOCHS:-5}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --trainable-scope "${TRAINABLE_SCOPE:-all}" \
  --head-lr "${HEAD_LR:-1e-4}" \
  --backbone-lr "${BACKBONE_LR:-1e-5}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --pos-weight-cap "${POS_WEIGHT_CAP:-10}" \
  --target-metric "${TARGET_METRIC:-macro_auroc}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-1000}" \
  "$@"
