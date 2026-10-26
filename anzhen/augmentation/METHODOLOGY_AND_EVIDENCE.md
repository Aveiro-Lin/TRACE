# Derivation methodology and evidence boundary

## Intended scope

Anzhen ECG Synthetic Augmentation v0.1.0 contains 396 label-preserving waveform variants generated from the clinician-verified real-patient training partition of Anzhen-ECG. The parent records correspond to genuine real-world clinical cases whose source clinical diagnoses underwent clinician verification. The derived variants are training augmentations, not new real-world cases, an independently sampled cohort, a source of new diagnoses, or independent clinical validation evidence. Verification of the real source cases is distinct from separate adjudication of the downstream rule-based PTB-XL/SCP mapping.

The derivation is reproducible from `build_synthetic_augmentation.py`, the locked split manifest recorded in `protocol.json`, the per-record random seeds and parameters in `synthetic_metadata.csv`, and the checksums in `SHA256SUMS.txt`.

## Evidence hierarchy

The cited sources play different roles and must not be conflated:

1. The AHA/ACCF/HRS statements provide standards for ECG acquisition, display, computerized processing, and diagnostic terminology [1,2,5]. They constrain how ECG signals and label semantics should be understood; they do not prescribe the synthetic transformation ranges used in this package.
2. PTB-XL provides the external data-model precedent for clinical 12-lead ECGs, machine-readable multi-label statements, patient-aware evaluation folds, and transparent metadata [3]. This package aligns a subset of its control-label vocabulary with the existing TRACE pipeline, but is not a reproduction of PTB-XL.
3. ECG augmentation research documents the use of baseline wander, Gaussian noise, magnitude scaling, and temporal displacement, while also showing that augmentation benefits are task-dependent and that morphology-changing operations may invalidate labels [4,6]. These sources support evaluating conservative perturbations; they do not establish clinical realism for any individual derived waveform.
4. The numerical ranges in v0.1.0 are implementation choices selected to remain small relative to standardized parent signals. Their use is evaluated empirically and through clinician review rather than attributed directly to a clinical guideline.

## Leakage-safe parent selection

1. The input split is fixed at the patient level before augmentation.
2. Only records whose locked split is `train` are eligible as parents. Validation and test signals are never read as parents.
3. The starting pool contains 541 real training records from 538 patients.
4. For each of ten PTB-XL-aligned report-control labels, the selection algorithm sets an effective minimum of 100 positive training instances unless the real training count already exceeds 100.
5. At each step, a real training record is sampled with probability proportional to its unresolved multi-label deficits and inversely related to prior use of the same patient. A patient may contribute at most five variants.
6. Because one parent may carry several labels, satisfying a rare-label deficit can also increase other label counts above 100. The target is therefore a minimum training-instance count, not an exactly balanced class distribution.
7. Selection stops once all requested minima are reached or another configured stop condition is met. Version 0.1.0 generated 396 variants from 161 parent patients and added zero independent patients.

## Signal transformation pipeline

Each variant is produced deterministically from its stored `sample_seed`. Operations are applied to a standardized 12-lead array of shape `(12, 2500)` at 250 Hz in the following order:

1. **Temporal displacement:** one integer offset is sampled uniformly from -25 to +25 samples (at most +/-100 ms) and applied identically to all 12 leads and to the validity mask. The current implementation uses a circular array shift.
2. **Per-lead gain:** each lead receives a multiplier sampled uniformly from 0.97 to 1.03.
3. **Low-frequency baseline perturbation:** each lead receives a sinusoid whose frequency is sampled uniformly from 0.10 to 0.35 Hz, phase from 0 to 2*pi, and amplitude from 0.005 to 0.025 of the standardized-signal scale.
4. **White noise:** zero-mean Gaussian noise is added independently per lead, with standard deviation sampled from 0.003 to 0.012 of the standardized-signal scale.
5. **Re-standardization:** every lead is re-centered and scaled to approximately zero mean and unit standard deviation.

All label columns are copied unchanged from the real training parent. No rule creates AFIB, QWAVE, ST depression, tachycardia, bradycardia, or any other diagnosis. The phrase “guideline-derived disease ECG” is therefore prohibited for this version.

Per-lead re-standardization largely removes a pure multiplicative gain change; consequently, the gain component should be viewed as part of the composite perturbation rather than as an independently persistent augmentation. This is a documented implementation property, not an additional clinical claim.

## Automated quality control

Generation fails if a variant contains non-finite values or a constant lead. A record passes the stored automated QC only when all of the following hold:

- maximum absolute residual lead mean is below `1e-4`;
- maximum absolute lead-standard-deviation error from 1 is below `1e-4`;
- minimum lead-wise Pearson correlation with the parent is at least 0.98 after undoing the known temporal shift;
- maximum absolute standardized amplitude is below 12;
- the provenance fields identify a training parent and the per-patient five-variant cap is respected.

All 396 records pass these numerical checks. These tests verify transformation integrity and high similarity to the parent and are distinct from the completed clinician review. Neither safeguard constitutes independent diagnostic relabelling or validation on an independent cohort.

## Known limitations and required interpretation

- The 396 variants are correlated with 161 real parent patients and do not increase the independent sample size.
- Class balancing is performed at the training-instance level. It does not estimate disease prevalence and must not be used for epidemiology.
- Parent labels are inherited rather than independently re-annotated. All included variants underwent clinician review and were accepted as label-preserving training augmentations, but this acceptance does not make them independent clinical observations.
- Circular shifting can introduce an array-boundary wrap artifact. Users should retain the released implementation for exact reproduction and treat crop/pad displacement without wrap-around as a prespecified future ablation.
- Baseline and noise components are low-amplitude simulations, not measurements of a named acquisition device or a patient-specific artifact process.
- Independent per-lead perturbations may not reproduce the full cross-lead covariance of real acquisition artifacts.
- The current package supports training only. Validation, threshold selection, confidence intervals, and final performance estimates must use real patients who were not augmentation parents.
- Clinician review is complete for all 396 included variants; its scope and interpretation boundary are recorded in this methodology and in the released package metadata.

## Recommended reporting language

> We generated 396 label-preserving training variants from 541 clinician-verified real-world training ECGs using small temporal displacement, per-lead gain, low-frequency baseline, and Gaussian-noise perturbations followed by per-lead re-standardization. All 396 included variants underwent clinician review and were accepted as label-preserving, training-only augmentations. Labels were inherited from the real training parent; the variants were excluded from validation and testing and were not counted as new clinical cases or independent patients.

Do not replace “training variants” with “patients,” “clinical cases,” or “newly generated disease ECGs.”

## References

1. Kligfield P, et al. Recommendations for the standardization and interpretation of the electrocardiogram: Part I. *Circulation*. 2007;115:1306-1324. doi:10.1161/CIRCULATIONAHA.106.180200.
2. Mason JW, et al. Recommendations for the standardization and interpretation of the electrocardiogram: Part II. *Heart Rhythm*. 2007;4:413-419. doi:10.1016/j.hrthm.2007.01.028.
3. Wagner P, et al. PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*. 2020;7:154. doi:10.1038/s41597-020-0495-6.
4. Raghu A, et al. Data Augmentation for Electrocardiograms. *Proceedings of Machine Learning Research*. 2022;174:282-310.
5. Wagner GS, et al. AHA/ACCF/HRS recommendations for the standardization and interpretation of the electrocardiogram: Part VI: acute ischemia/infarction. *Circulation*. 2009;119:e262-e270. doi:10.1161/CIRCULATIONAHA.108.191098.
6. Rahman MM, et al. A Systematic Survey of Data Augmentation of ECG Signals for AI Applications. *Sensors*. 2023;23:5237. doi:10.3390/s23115237.

Machine-readable entries are provided in `REFERENCES.bib`.
