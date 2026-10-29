# ME 3310 — Motion Classification (MATLAB)

An open educational MATLAB workspace for **IMU-based signal processing, feature extraction, and motion classification**, supporting **Experimental Methods I (ME 3310)**.

The central engineering question is:

> Can we distinguish a **target motion** from a **confusing motion**, and how much evidence supports the decision?

**Engineering Question → Measurement → Evidence → Analysis/Inference → Engineering Judgment**

## Quick start

Use MATLAB **R2023b or newer** if you want to use MATLAB's built-in Git commands:

```matlab
repo = gitclone("https://github.com/stargaze221/ME3310-Motion-Classification.git");
cd(repo.WorkingFolder)
```

Then, from the repository root, run the example scripts **in order**:

```matlab
run("examples/USC_HAD/step01_prepare_up_down.m")
run("examples/USC_HAD/step02_preview_up_down.m")
run("examples/USC_HAD/step02b_explore_full_psd.m")
run("examples/USC_HAD/step03_extract_window_features.m")
run("examples/USC_HAD/step04_train_linear_svm.m")
run("examples/USC_HAD/step05_compare_power_ratio.m")
```

Already cloned the repository? From the clone's root directory, update it with `repo = gitrepo; pull(repo);` instead of cloning again. If MATLAB's Git functions are unavailable, use a normal Git client or download the repository ZIP.

**If you already prepared the dataset on your computer**, copy `ME3310_USC_HAD_UpDown.mat` into `examples/USC_HAD/`, then begin with script 02. To reuse a previously downloaded official archive, edit `sourceFolder` in script 01 to the folder holding `USC-HAD.zip` or `USC_HAD_raw`.

## Example: Upstairs versus Downstairs

The [USC-HAD example](examples/USC_HAD/README.md) uses official wearable IMU recordings (100 Hz, three acceleration axes and three gyroscope axes) for two similar motions:

- **Walking Upstairs**: binary label 0 (USC-HAD activity 4)
- **Walking Downstairs**: binary label 1 (USC-HAD activity 5)

| Script | What you see or do |
|---|---|
| [step01_prepare_up_down.m](examples/USC_HAD/step01_prepare_up_down.m) | Download/reuse the official archive, prepare original complete trials |
| [step02_preview_up_down.m](examples/USC_HAD/step02_preview_up_down.m) | Raw IMU signals, acceleration magnitude, Welch power spectral density |
| [step02b_explore_full_psd.m](examples/USC_HAD/step02b_explore_full_psd.m) | Full-frequency PSD, normalized spectral shape, and subject-by-subject band comparison (training subjects only) |
| [step03_extract_window_features.m](examples/USC_HAD/step03_extract_window_features.m) | Sliding windows, time- and frequency-domain features, scatter plot |
| [step04_train_linear_svm.m](examples/USC_HAD/step04_train_linear_svm.m) | Two-feature linear SVM, decision boundary, held-out-subject evaluation |
| [step05_compare_power_ratio.m](examples/USC_HAD/step05_compare_power_ratio.m) | Compare STD, absolute PSD band powers, and log high/low power ratio with subject-wise CV |

**Follow-up power-ratio experiment:** Run script 05 after script 01; it independently extracts the spectral features from the prepared USC-HAD data. It reports leave-one-training-subject-out cross-validation for four candidates and, optionally, a separately labeled diagnostic on the four already inspected subjects. Such a reused holdout is NOT a fresh test.

Inspect the full PSD before choosing a band. The initial 0.8-3 Hz band in script 03 is only a demonstration; change it based on training-subject evidence and preserve held-out subjects for validation.

**MATLAB requirements:** base MATLAB; Signal Processing Toolbox for `pwelch`; Statistics and Machine Learning Toolbox for `fitcsvm`. The new script has not yet been runtime-verified in a MATLAB installation.

## Why this is an Experimental Methods project

A machine-learning classifier can be trained with very few distinct experiments, but its apparent accuracy may be misleading. Overlapping windows from a single IMU trial are **not independent replications**. These examples preserve subject/trial identity and reserve entire subjects for testing instead of randomly mixing nearly duplicate windows across training and test sets.

For ME 3310 Project Week 2, instructors can demonstrate the common USC-HAD data and then ask teams to transfer the analysis workflow to their own **Week 1 target motion versus confusing motion** measurements. Week 3 can investigate uncertainty, motion transitions, and validation on independently recorded trials.

## Data rights and attribution

**No USC-HAD raw or processed sensor data are included in this public repository.** The scripts download data from the [official USC-SIPI USC-HAD website](https://sipi.usc.edu/had/). Check the original provider's terms before redistributing any copies.

Source reference: M. Zhang and A. A. Sawchuk, *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors*, SAGAware Workshop, 2012.

Code is shared here as course material; no third-party data or license terms are implied by this README.
