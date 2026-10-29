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
run("examples/USC_HAD/step06_compare_dnn.m")
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
| [step03_extract_window_features.m](examples/USC_HAD/step03_extract_window_features.m) | Sliding windows with 8 original features plus 8 additional PSD band powers (16 total) |
| [step04_train_linear_svm.m](examples/USC_HAD/step04_train_linear_svm.m) | Two-feature linear SVM, decision boundary, held-out-subject evaluation |
| [step05_compare_power_ratio.m](examples/USC_HAD/step05_compare_power_ratio.m) | Compare STD, absolute PSD band powers, and log high/low power ratio with subject-wise CV |
| [step06_compare_dnn.m](examples/USC_HAD/step06_compare_dnn.m) | Compare five SVM/DNN models at both 3-second window and entire-trial (trajectory) levels |
| [step07_prepare_five_motions.m](examples/USC_HAD/step07_prepare_five_motions.m) | Reuse the USC-HAD archive to prepare five verified classes, retaining original trials |
| [step08_extract_five_motion_features.m](examples/USC_HAD/step08_extract_five_motion_features.m) | Reuse identical 16 IMU features with numeric class labels 0-4 |
| [step09_train_five_motion_svm.m](examples/USC_HAD/step09_train_five_motion_svm.m) | Multiclass one-versus-one linear SVM, LOSO CV, window and trial confusion matrices |
| [step10_demo_five_motion_sequence.m](examples/USC_HAD/step10_demo_five_motion_sequence.m) | Explicitly artificial montage of five-motion predictions and simple causal smoothing |

## Multi-motion extension (five classes)

USC-HAD activity IDs **1, 4, 5, 6, 9** map to labels **0-4**: Walking Forward, Upstairs, Downstairs, Running Forward and Standing. The original two-motion examples above are preserved.

Run these scripts after updating your checkout; step07 reuses the already-downloaded official data when available:

```matlab
repo = gitrepo;
pull(repo);
run("examples/USC_HAD/step07_prepare_five_motions.m")
run("examples/USC_HAD/step08_extract_five_motion_features.m")
run("examples/USC_HAD/step09_train_five_motion_svm.m")
run("examples/USC_HAD/step10_demo_five_motion_sequence.m") % Optional illustration
```

Script 09 compares five-class ECOC linear SVMs with **8 and 16 features**, evaluates **five-class macro recall** (balanced accuracy) separately for overlapping windows and original trial majority votes, and reports confusion matrices. It holds out entire subjects for each training-group CV fold. The four previously inspected subjects are explicitly labeled as a diagnostic, not an untouched test. Script 10 concatenates **window features from separate trials**, NOT physical sensor streams: it is only a pseudo-stream illustration of temporal smoothing, with artificial join boundaries. A genuine detector requires a continuously measured and time-annotated activity sequence.

**Follow-up power-ratio experiment:** Run script 05 after script 01; it independently extracts the spectral features from the prepared USC-HAD data. It reports leave-one-training-subject-out cross-validation for four candidates and, optionally, a separately labeled diagnostic on the four already inspected subjects. Such a reused holdout is NOT a fresh test.

**Multi-subject neural network experiment:** Rerun the UPDATED script 03 to regenerate `ME3310_USC_HAD_WindowFeatures.mat` with **16 features**, then run script 06. It now compares five configurations: linear SVM with 2, 8, or 16 features, and a two-hidden-layer `fitcnet` neural network with 8 or 16 features. It reports **both window-level and trajectory-level** leave-one-training-subject-out balanced accuracy. Original trials are grouped by `recordID`, with one majority-vote decision per trial. Exact window-vote ties use the classifier score average. Each subject contributes equally to the primary cross-validation summary. The four previously inspected subjects remain a diagnostic, not a new untouched test. Added acceleration magnitude PSD bands: **0.3-0.8, 0.8-1.5, 1.5-3, 3-5, 5-8, 8-12, 12-20, 20-40 Hz**. The original 0.8-3 Hz band (feature 2) remains unchanged, so some PSD features are correlated.

Inspect the full PSD before choosing a band. The initial 0.8-3 Hz band in script 03 is only a demonstration; change it based on training-subject evidence and preserve held-out subjects for validation.

**Interpretation:** A trajectory is one original, single-activity USC-HAD recording; trajectory accuracy measures whole-trial classification, not online motion onset/transition detection. Window and trajectory metrics may differ, and correlated windows cannot be counted as independent trials.

**MATLAB requirements:** base MATLAB; Signal Processing Toolbox for `pwelch`; Statistics and Machine Learning Toolbox for `fitcsvm`. The new script has not yet been runtime-verified in a MATLAB installation.

## Student-measured IMU files (MAT format)

[Student data inspection tools](examples/Student_IMU/README.md) support actual ME 3310 MAT trial files with elapsed time and optional 3-axis acceleration and gyroscope measurements. Use [step01_inspect_student_mat.m](examples/Student_IMU/step01_inspect_student_mat.m) to select local recordings, inspect sampling, channels, time traces and PSD. [load_student_trial.m](examples/Student_IMU/load_student_trial.m) returns a standard trial structure without inventing unavailable sensor channels or class labels. The initial uploaded examples are **accelerometer-only at 20 Hz**, requiring a different feature set from the USC-HAD 100-Hz six-axis model. No student data are committed.

## Week 11 lecture: Time series, FFT and PSD (Page 01)

The [Week 11 student-facing MATLAB demo](examples/Student_IMU/week11_01_feature_demo.m) supports the Canvas page **01 — From Time Series to Meaningful Features**. It uses 20-Hz labeled **synthetic** six-axis signals by default, or reads a real local MAT trial via the existing loader. Each subsection has a matching function call:

```matlab
repo = gitrepo;
pull(repo);
addpath("examples/Student_IMU");
week11_01_feature_demo("signal");   % raw accelerometer/gyro axes
R = week11_01_feature_demo("time");   % mean, STD, RMS, peak, jerk
R = week11_01_feature_demo("fft");    % FFT magnitude versus PSD
R = week11_01_feature_demo("psd");    % Welch versus periodogram
R = week11_01_feature_demo("bands");  % 0.5-3 and 3-8 Hz power
R = week11_01_feature_demo("windows");% overlapping window dependence
R = week11_01_feature_demo("all","C:\\path\\to\\trial_T01.mat");
```

The real T01/T02 files previously available were **acceleration only**; the script does not fabricate gyro readings. For next student measurements, collect acceleration XYZ + gyroscope XYZ at nominal 20 Hz and verify channels and units. Use **three independent Target and three independent Confusing trials** for repeatability; do not treat overlapping windows as independent experiments. A MATLAB installation and Signal Processing Toolbox (for PSD) are needed. Code has been reviewed statically, not executed in a MATLAB runtime.

## Why this is an Experimental Methods project

A machine-learning classifier can be trained with very few distinct experiments, but its apparent accuracy may be misleading. Overlapping windows from a single IMU trial are **not independent replications**. These examples preserve subject/trial identity and reserve entire subjects for testing instead of randomly mixing nearly duplicate windows across training and test sets.

For ME 3310 Project Week 2, instructors can demonstrate the common USC-HAD data and then ask teams to transfer the analysis workflow to their own **Week 1 target motion versus confusing motion** measurements. Week 3 can investigate uncertainty, motion transitions, and validation on independently recorded trials.

## Data rights and attribution

**No USC-HAD raw or processed sensor data are included in this public repository.** The scripts download data from the [official USC-SIPI USC-HAD website](https://sipi.usc.edu/had/). Check the original provider's terms before redistributing any copies.

Source reference: M. Zhang and A. A. Sawchuk, *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors*, SAGAware Workshop, 2012.

Code is shared here as course material; no third-party data or license terms are implied by this README.
