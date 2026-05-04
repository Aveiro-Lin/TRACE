# Checkpoint Directory

Dataset-specific TRACE weights are not redistributed in this open-source release. The PTB-XL, CPSC2018, and Ningbo subdirectories are empty output skeletons that are populated by the training scripts.

The only retained pretrained file is:

```text
ckpt/pretrained_weight/CoRe-ECG_mimic_encoder.pth
```

This file initializes the TRACE ECG encoder from the CoRe-ECG mimic pretrained checkpoint. Re-run the dataset workflows in the repository `README.md` to produce tokenizer, diagnosis, report-control, LoRA, and merged report-model checkpoints.

