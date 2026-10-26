# TRACE release manifest

This directory is the current TRACE code release and contains the maintained public workflows plus the Anzhen data/adaptation pipeline.

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

- Anzhen:
  - locked external diagnosis evaluation: `evaluate_anzhen_external.py`
  - patient-level diagnosis adaptation: `finetune_anzhen_diagnosis.py`
  - patient-level report-control adaptation: `finetune_anzhen_report_control.py`
  - real-only and real-plus-derived launchers: `scripts/run_anzhen_*.sh`
  - de-identification/package construction and release validation: `anzhen/processing/`
  - train-only derived augmentation construction: `anzhen/augmentation/`

## Checkpoints

Model weights are not redistributed with the open-source release, except for the small public initialization used by the TRACE ECG encoder:

- `ckpt/pretrained_weight/CoRe-ECG_mimic_encoder.pth`: CoRe-ECG MIMIC-IV-ECG-pretrained encoder initialization ([arXiv:2604.11359](https://arxiv.org/abs/2604.11359)); SHA-256 `24472f7ab730a49054ebcf6bce43c163da37f376bcfb0e6fadbf28c4fd86c110`.
- `ckpt/ptbxl`, `ckpt/cpsc2018`, `ckpt/ningbo`, and optional `ckpt/anzhen`: checkpoint-output locations. Re-run the training scripts to populate tokenizer, diagnosis, report-control, LoRA, and merged report-model weights.

All default checkpoint paths point directly into `ckpt/`; no compatibility weight directories are required at the repository root.

Raw waveform arrays, restricted source material, derived waveform arrays, private data, logs, bytecode caches, and older non-selected experiment folders are intentionally excluded. The code preserves the paper's patient-disjoint 40:10:50 Anzhen protocol and restricts the 396 reviewed derived variants to training only.
