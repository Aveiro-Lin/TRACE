# Data directory

This directory intentionally keeps only small TRACE-specific configuration or split files. Public dataset files that can be downloaded from the original providers are not redistributed here.

## Files kept in this repository

- `class.json`: TRACE report/control option vocabulary used by the report-template utilities.
- `TRAIN_reference_cpsc2018.csv`, `VAL_reference_cpsc2018.csv`, `TEST_reference_cpsc2018.csv`: TRACE train/validation/test split tables for CPSC2018. These files contain record identifiers and label indices derived from the public CPSC2018 labels, but do not contain ECG waveforms.
- `ningbo_label_names_25.json`: label-name mapping used by the Ningbo scripts to interpret the local label-vector order.
- `ptbxl_validation_thresholds.json`: thresholds selected on PTB-XL validation data and applied unchanged in the optional Anzhen zero-shot analysis. The primary fixed threshold remains 0.5.

## Files users must download or prepare

- PTB-XL metadata: download the official PTB-XL release and place `ptbxl_database.csv` and `scp_statements.csv` in this directory, or pass their locations with `--metadata-csv` and `--scp-statements-csv`.
- ECG waveform files: download or prepare the waveform records required by PTB-XL, CPSC2018, and Ningbo, then pass the corresponding dataset root paths to the training/evaluation scripts.
- Anzhen package: obtain the separately released de-identified package and pass its root with `--data-root` or `ANZHEN_DATA_ROOT`. TRACE consumes the following aligned files:

  ```text
  trace_compat/
    full_data.npy
    valid_mask.npy
    ptbxl_compat_database.csv
    evaluation_subsets.csv
    row_index.csv
    paper_report_control_split_seed42.csv
  ```

  The real dataset and the reviewed derived augmentation package remain separate. The latter is supplied only to `finetune_anzhen_report_control.py --derived-root` and may augment the training partition only.

Generated reports, cached control prompts, metric JSON files, figure intermediates, large waveform arrays, and dataset archives are intentionally excluded from the release staging copy.
