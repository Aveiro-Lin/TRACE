# Anzhen data and TRACE experiment workflow

This directory contains code and documentation only. It does not contain ECG arrays, source clinical files, patient identifiers, model weights, or experiment outputs.

## Data roles

The separately managed real-data package contains 1,385 de-identified ECG records from 1,377 patients in a PTB-XL-compatible representation. These are genuine real-world clinical cases whose source diagnoses underwent clinical verification. The compatibility mapping into the local PTB-XL/SCP vocabulary is a deterministic interoperability layer and should not be conflated with the underlying clinical verification.

The paper's report-control cohort contains 1,352 records with at least one usable diagnosis, form, or rhythm target. Its locked patient-level allocation is:

| Partition | Real records | Patients | Role |
|---|---:|---:|---|
| Train | 541 | 538 | adaptation and augmentation parents |
| Validation | 134 | 134 | model selection and thresholds |
| Test | 677 | 673 | locked final evaluation |
| Excluded | 33 | 32 | no usable report-control target |

The 50:50 protocol first reserves half of eligible patients for testing, then divides the development half 80:20, yielding approximately 40:10:50 overall. Patients do not overlap across train, validation, and test.

The separate augmentation package contains 396 label-preserving variants generated only from real training parents. All included variants underwent clinician review and were accepted for training-only augmentation. They inherit their parent's patient identity and labels; they are not new patients, new acquisitions, independent observations, validation data, or test data.

## Expected TRACE-compatible files

```text
ANZHEN_DATA_ROOT/
  anzhen_database.csv
  anzhen_statements.csv
  trace_compat/
    full_data.npy
    valid_mask.npy
    ptbxl_compat_database.csv
    evaluation_subsets.csv
    row_index.csv
    scp_statements.csv
    paper_report_control_split_seed42.csv
```

The aligned TRACE arrays have shape `(record, 12, 2500)` with the fixed lead order `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6`. The metadata, subset table, row-index table, signal array, and validity mask must retain identical row order.

The derived package consumed by `finetune_anzhen_report_control.py --derived-root` contains:

```text
ANZHEN_DERIVED_ROOT/
  synthetic_signals.npy
  synthetic_valid_mask.npy
  synthetic_metadata.csv
  protocol.json
```

The filenames retain the original experiment schema for reproducibility; in manuscript-facing language these records are called clinician-reviewed, label-preserving derived variants.

## Code map

- `processing/build_dataset.py`: private end-to-end dataset construction and de-identification pipeline.
- `processing/export_paper_report_control_split.py`: exports the locked paper split into the release metadata.
- `processing/verify_release.py`: full internal integrity, privacy, WFDB, TRACE-array, and split audit. It requires the restricted private mapping and is not needed for model inference.
- `processing/finalize_manifest.py`: regenerates the release manifest and SHA-256 list.
- `augmentation/build_synthetic_augmentation.py`: deterministically generates label-preserving train-only variants from the locked real training partition.
- `augmentation/finalize_manifest.py`: regenerates the augmentation-package manifest and checksums.
- `augmentation/METHODOLOGY_AND_EVIDENCE.md`: transformation, evidence, quality-control, and interpretation boundaries.

## Internal package construction

The full builder operates only on restricted local source material and writes private linkage artifacts to a separate protected directory:

```bash
python anzhen/processing/build_dataset.py \
  --input-root /restricted/source/root \
  --output-root /path/to/Anzhen-release \
  --private-root /restricted/private-audit-root \
  --ptbxl-root /path/to/PTB-XL
```

Run the full internal release audit before packaging:

```bash
python anzhen/processing/verify_release.py \
  --public-root /path/to/Anzhen-release \
  --private-root /restricted/private-audit-root
```

Regenerate manifests only after all approved files are final:

```bash
python anzhen/processing/finalize_manifest.py --root /path/to/Anzhen-release
python anzhen/augmentation/finalize_manifest.py --root /path/to/Anzhen-derived-augmentation
```

To regenerate the derived training package, first produce the enriched split manifest without training, then pass it to the builder explicitly:

```bash
python finetune_anzhen_report_control.py \
  --data-root /path/to/Anzhen-release \
  --source-checkpoint /path/to/ptbxl/report_control/best.pt \
  --output-dir results/anzhen_report_control/split_50_50_seed42 \
  --split-only

python anzhen/augmentation/build_synthetic_augmentation.py \
  --real-root /path/to/Anzhen-release \
  --split-manifest results/anzhen_report_control/split_50_50_seed42/split_manifest.csv \
  --output-dir /path/to/Anzhen-derived-augmentation
```

Do not copy raw source files, linkage keys, source-to-public mappings, protected audit records, patient data, or trained weights into this repository.

## Model experiments

From the repository root:

```bash
export ANZHEN_DATA_ROOT=/path/to/Anzhen-release
export ANZHEN_DERIVED_ROOT=/path/to/Anzhen-derived-augmentation

scripts/run_anzhen_finetune_diagnosis_50_50.sh
scripts/run_anzhen_report_control_50_50.sh
scripts/run_anzhen_report_control_derived_50_50.sh
```

The frozen `trace_compat/paper_report_control_split_seed42.csv` in the data package is the authoritative report-control split. The report-control script validates its row and public-ID alignment, verifies parent membership, and refuses derived records whose parent is not in the real training partition. It does not re-search a replacement split. Validation and test metrics are computed exclusively on real patients.

The optional external evaluator is retained for reproducibility:

```bash
scripts/run_anzhen_external_ptbxl.sh
```

It performs inference only and does not use Anzhen records for optimization, checkpoint selection, or threshold selection.
