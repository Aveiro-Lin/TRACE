# TRACE release staging manifest

This directory is a trimmed staging copy of the working TRACE codebase for TRACE open-source preparation.

## Dataset workflows retained

- PTB-XL:
  - tokenizer training: `train_tokenizer_ptbxl.py`
  - report-control head training: `report_control_train_ptbxl.py`
  - diagnosis checkpoint construction: `train_diagnosis_ckpt.py --dataset ptbxl`
  - diagnosis evaluation: `eval_diagnosis_ptbxl.py`
  - report training: `train_report_ptbxl.py`
  - report inference: `infer_report_ptbxl.py`
  - report metrics: `eval_ptb_report_generation_metrics.py`

- CPSC2018:
  - tokenizer training: `train_tokenizer_cpsc2018.py`
  - report-control head training: `report_control_train_cpsc2018.py`
  - diagnosis checkpoint construction: `train_diagnosis_ckpt.py --dataset cpsc2018`
  - diagnosis evaluation: `eval_diagnosis_cpsc2018.py`
  - report training: `train_report_cpsc2018.py`
  - structured report/control inference: `infer_report_cpsc2018.py`

- Ningbo:
  - tokenizer training: `train_tokenizer_ningbo.py`
  - report-control head training: `report_control_train_Ningbo.py`
  - diagnosis checkpoint construction: `train_diagnosis_ckpt_ningbo.py`
  - diagnosis evaluation: `eval_diagnosis_ningbo.py`
  - report training: `train_report_ningbo.py`
  - structured report/control inference: `infer_report_ningbo.py`

## Checkpoints

Model weights are not redistributed with the open-source release, except for the small public initialization used by the TRACE ECG encoder:

- `ckpt/pretrained_weight/CoRe-ECG_mimic_encoder.pth`: CoRe-ECG mimic pretrained encoder initialization.
- `ckpt/ptbxl`, `ckpt/cpsc2018`, and `ckpt/ningbo`: empty checkpoint-output skeletons. Re-run the training scripts to populate tokenizer, diagnosis, report-control, LoRA, and merged report-model weights.

All default checkpoint paths point directly into `ckpt/`; no compatibility weight directories are required at the repository root.

Raw waveform arrays, parquet files, private data, logs, bytecode caches, and older non-selected experiment folders are intentionally excluded.
