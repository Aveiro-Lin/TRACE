# Checkpoint Directory

Dataset-specific TRACE weights are not redistributed in this open-source release. The PTB-XL, CPSC2018, and Ningbo subdirectories are empty output skeletons that are populated by the training scripts.

The only retained pretrained file is:

```text
ckpt/pretrained_weight/CoRe-ECG_mimic_encoder.pth
```

This file initializes the TRACE ECG encoder from the MIMIC-IV-ECG-pretrained
checkpoint described in:

> Qin Z, Lin X, Zhang P, et al. CoRe-ECG: Advancing Self-Supervised
> Representation Learning for 12-Lead ECG via Contrastive and Reconstructive
> Synergy. arXiv:2604.11359 (2026).
> https://arxiv.org/abs/2604.11359

Integrity information:

```text
SHA-256  24472f7ab730a49054ebcf6bce43c163da37f376bcfb0e6fadbf28c4fd86c110
File     ckpt/pretrained_weight/CoRe-ECG_mimic_encoder.pth
```

The TRACE authors distribute this bundled checkpoint with the repository under
the MIT License. This grant does not apply to datasets, language-model assets,
or other third-party materials that are not included in this repository.

Re-run the dataset workflows in the repository `README.md` to produce tokenizer,
diagnosis, report-control, LoRA, and merged report-model checkpoints.

`ckpt/anzhen/` is an empty output skeleton for the patient-level adapted diagnosis and report-control checkpoints. Anzhen data and weights are not bundled with the source repository.
