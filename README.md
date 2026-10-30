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

## Week 11 lecture: REAL USC-HAD time series, FFT, and PSD (Page 01)

**Whole trial versus the first measured sliding window (Page 01, Section 3):**
The `"time"` mode computes Mean, sample STD, RMS, peak and jerk using
**every sample in a single original trial**. It does not compute the
first 3-s analysis window. The new `"timecompare"` mode makes a
direct, apples-to-apples comparison of the **whole original trial**
against the **first 3 seconds (300 samples at USC-HAD's 100 Hz)**
and displays a comparison table and a figure.

```matlab
repo = gitrepo; pull(repo);  % from your existing repository root
addpath(fullfile(pwd,"examples","Student_IMU"));
R = week11_01_feature_demo("time","upstairs");
disp(R.timeFeatures)               % WHOLE original trial
C = week11_01_feature_demo("timecompare","upstairs");
disp(C.timeComparison)              % WHOLE versus FIRST 3-s window
D = week11_01_feature_demo("timecompare","downstairs");
disp(D.timeComparison)
```

This demonstration analyzes only measured USC-HAD samples, without
generating data. Both summaries refer to the SAME original trial
and therefore are not independent experimental repetitions. Run
`week11_01_feature_demo("windows","upstairs")` to examine multiple
overlapping 3-s windows with 0.5-s hop. No classifier or feature
selection is performed by the comparison command.


### Compare two real motions in feature space (Week 11 Page 01, Section 8)

### Week 11 Canvas Page 02 — Feature Comparison, Fisher Score and LDA

### Week 11 Canvas Page 05 — Uncertainty and Reliability

The [Page 05 MATLAB demonstration](examples/Student_IMU/week11_05_uncertainty_demo.m)
uses **actual USC-HAD 100-Hz original recordings and previously saved five-class SVM results**.
It introduces (a) uncertainty in a mean *measured feature* from independent
original trials, (b) differences across leave-one-subject-out folds, and
(c) five-class original-trial confusion and per-class recall. No artificial
measurements, new classifier fit, or automatic optimization.

From the updated repository root, after running Page 04's step07, step08,
and step09, run:

```matlab
addpath(fullfile(pwd,"examples","Student_IMU"));
U = week11_05_uncertainty_demo("all");
disp(U.featureIntervals);
disp(U.subjectReport);
disp(U.classRecall);
```

Or run the individual sections: `"features"`, `"subjects"`, `"errors"`.
The student t interval is for the **mean of independent trial-level
feature measurements for one subject**, not classification accuracy.
Cross-validation fold scores share training subjects; a naive t-based
confidence interval across those folds is not a valid model reliability
guarantee. Previously viewed diagnostic subjects are not a fresh final
test. Source data must exist locally and Statistics and Machine Learning
Toolbox is required for the confidence interval. MATLAB execution
verification is pending.

### Week 11 Canvas Page 03 — From Features to Machine Learning

[week11_03_classifiers_demo.m](examples/Student_IMU/week11_03_classifiers_demo.m) uses **real USC-HAD Upstairs/Downstairs original trials** to compare logistic regression (`fitglm`) and a linear SVM (`fitcsvm`). It calculates the same 16 features previously extracted from overlapping windows but averages them **within each original trial before training**. One original measured trial is one observation; this is **completed-trial classification**, NOT real-time transition detection.

From an updated repository root (after step01 and step03 data preparation):

```matlab
addpath(fullfile(pwd,"examples","Student_IMU"));
M = week11_03_classifiers_demo("manual",[1 2]);
disp(M.trainingReport);
disp(M.diagnosticReport);
M.confusionLogistic
M.confusionSVM
```

Plots: (1) logistic probability 0.5 and SVM margin 0 decision boundaries plus actual trial points, and (2) confusion matrices for subjects excluded from the model fit. Feature normalization uses development data only. Development subjects: `[2 3 4 7 8 9 10 11 12 14]`; diagnostic subjects: `[1 5 6 13]`. These diagnostic subjects had already been examined earlier in the project, so this is **NOT** an untouched final test.

Optional limited selection among six predefined feature pairs, using development-only leave-one-subject-out cross-validation:

```matlab
M = week11_03_classifiers_demo("search");
disp(M.candidateResults);
disp(M.featurePair);
```

This screening is exploratory and does not optimize SVM hyperparameters. SVM margins are not probabilities; logistic probabilities also require validation/calibration. Requires Statistics and Machine Learning Toolbox; existing PSD extraction requires Signal Processing Toolbox. The script was inspected statically but not executed in MATLAB.


[week11_02_fisher_lda_demo.m](examples/Student_IMU/week11_02_fisher_lda_demo.m) reuses the **measured original-trial feature table** returned by the comparison demo. It computes descriptive, trial-level **Fisher Scores**, displays feature-redundancy correlations, and shows a two-feature **regularized LDA projection** (not a fitted/tested classifier). No simulated data or automatic model optimization.

After preparing the real USC-HAD recordings and window-feature file as above:

```matlab
addpath(fullfile(pwd,"examples","Student_IMU"));
C = week11_02_feature_compare_demo(2,[1 2]);
D = week11_02_fisher_lda_demo(C,[1 2]);
disp(D.fisherTable);
D = week11_02_fisher_lda_demo(C,[1 3]);
```

Subject 2 yields five original trials per class when available. Windows are first summarized within each trial, never treated as independent experimental replicates. Scores and LDA direction are explored **on the same observed data**; they are *not* estimates of prediction accuracy on a future trial or another participant. This script uses Base MATLAB for Fisher/LDA and expects Signal Processing Toolbox for the earlier feature extraction/comparison step. MATLAB runtime testing is still pending.


The [USC-HAD trial-level comparison demo](examples/Student_IMU/week11_02_feature_compare_demo.m) visualizes Upstairs vs Downstairs using the existing **16-feature** extraction pipeline. It creates (1) two 1D feature comparisons, (2) a 2D scatter, and (3) an equally trial-weighted PSD comparison. Each dot represents **one original trial**, not one overlapping window.

After cloning or pulling the repository, from its root:

```matlab
if ~isfile("examples/USC_HAD/ME3310_USC_HAD_UpDown.mat")
    run("examples/USC_HAD/step01_prepare_up_down.m");
end
if ~isfile("examples/USC_HAD/ME3310_USC_HAD_WindowFeatures.mat")
    run("examples/USC_HAD/step03_extract_window_features.m");
end
addpath(fullfile(pwd,"examples","Student_IMU"));
C = week11_02_feature_compare_demo(2,[1 2]);
disp(C.trialFeatures)
C = week11_02_feature_compare_demo(2,[1 3]);
C = week11_02_feature_compare_demo("all",[1 2]);
```

The default is Subject 2, comparing 5 real upstairs and 5 real downstairs trials. A trial's feature vector is the mean of its own sliding-window vectors; plots are *descriptive*, not classification accuracy, and do not imply independent sliding windows. PSDs are averaged window-to-trial-to-class with equal weight per original trial, using the same Welch settings as the original feature-extraction script. Signal Processing Toolbox is required. MATLAB execution has not yet been independently verified.


The [Week 11 MATLAB demonstration](examples/Student_IMU/week11_01_feature_demo.m) now uses **original USC-HAD measured 100-Hz six-axis recordings** as the default data source. It compares Walking Upstairs and Walking Downstairs for Subject 2, Trial 1 when available. The original measured trials are prepared through [step01_prepare_up_down.m](examples/USC_HAD/step01_prepare_up_down.m), and are **not stored in this repository**.

**First time** (MATLAB R2023b or newer, from a folder where you want to clone):

```matlab
repo = gitclone("https://github.com/stargaze221/ME3310-Motion-Classification.git");
cd(repo.WorkingFolder);
run("examples/USC_HAD/step01_prepare_up_down.m"); % official dataset, once
addpath(fullfile(pwd,"examples","Student_IMU"));
week11_01_feature_demo("signal","upstairs");
```

**Already cloned** (from the existing repository folder):

```matlab
repo = gitrepo;
pull(repo);
if ~isfile("examples/USC_HAD/ME3310_USC_HAD_UpDown.mat")
    run("examples/USC_HAD/step01_prepare_up_down.m");
end
addpath(fullfile(pwd,"examples","Student_IMU"));
R = week11_01_feature_demo("time","upstairs");
R = week11_01_feature_demo("fft","upstairs");
R = week11_01_feature_demo("psd","downstairs");
R = week11_01_feature_demo("bands","upstairs");
R = week11_01_feature_demo("windows","upstairs");
```

The function also accepts an actual student MAT-file path as its second argument. **USC-HAD is 100 Hz (Nyquist 50 Hz), whereas planned PocketLab recordings are 20 Hz (Nyquist 10 Hz)**. The introductory 0.5–3 and 3–8 Hz exploratory PSD bands are accessible in both. For next student measurements, collect acceleration XYZ + gyroscope XYZ, verify units/timestamps and use three independent Target and three independent Confusing trials; overlapping windows are not independent repetitions. Signal Processing Toolbox is required for Welch PSD, and the code has been reviewed but not executed in MATLAB.


## Why this is an Experimental Methods project

A machine-learning classifier can be trained with very few distinct experiments, but its apparent accuracy may be misleading. Overlapping windows from a single IMU trial are **not independent replications**. These examples preserve subject/trial identity and reserve entire subjects for testing instead of randomly mixing nearly duplicate windows across training and test sets.

For ME 3310 Project Week 2, instructors can demonstrate the common USC-HAD data and then ask teams to transfer the analysis workflow to their own **Week 1 target motion versus confusing motion** measurements. Week 3 can investigate uncertainty, motion transitions, and validation on independently recorded trials.

## Data rights and attribution

**No USC-HAD raw or processed sensor data are included in this public repository.** The scripts download data from the [official USC-SIPI USC-HAD website](https://sipi.usc.edu/had/). Check the original provider's terms before redistributing any copies.

Source reference: M. Zhang and A. A. Sawchuk, *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors*, SAGAware Workshop, 2012.

Code is shared here as course material; no third-party data or license terms are implied by this README.
