# ME 3310 Project Week 2 — USC-HAD: Upstairs vs. Downstairs

**Course goal:** Engineering Question → Measurement → Evidence → Analysis / Inference → Engineering Judgment.

Real 100-Hz six-axis IMU data are used to distinguish **Walking Upstairs** from **Walking Downstairs**. This demonstration supports the student teams' Week 1 **Target Motion vs. Confusing Motion** investigations. Student teams should later transfer the tools to **their own data and motion definitions**.

## Run the examples in MATLAB

From the root of your [ME3310-Motion-Classification](https://github.com/stargaze221/ME3310-Motion-Classification) checkout, on branch **main**:

~~~matlab
repo = gitrepo;
pull(repo);
run("examples/USC_HAD/step01_prepare_up_down.m");
run("examples/USC_HAD/step02_preview_up_down.m");
run("examples/USC_HAD/step02b_explore_full_psd.m");
run("examples/USC_HAD/step03_extract_window_features.m");
run("examples/USC_HAD/step04_train_linear_svm.m");
run("examples/USC_HAD/step05_compare_power_ratio.m");
run("examples/USC_HAD/step06_compare_dnn.m");
~~~

Or open the files in the MATLAB Editor and run them individually in order.

**Already downloaded USC-HAD?** In script 01 set the variable <code>sourceFolder</code> to the directory containing <code>USC-HAD.zip</code> or <code>USC_HAD_raw</code>. The processed MAT file is saved beside the scripts. If you already have <code>ME3310_USC_HAD_UpDown.mat</code> from the previous starter, copy it into this folder and skip script 01.

## Files

| File | Purpose |
|---|---|
| step01_prepare_up_down.m | Obtain official USC-HAD and extract **all original activity-4 and activity-5 trials** |
| step02_preview_up_down.m | Plot raw accelerometer/gyroscope, magnitude and complete-trial Welch PSD |
| step02b_explore_full_psd.m | Full 0-50 Hz PSD, 0-10 Hz zoom, normalized PSD shape and subject-by-subject band consistency |
| step03_extract_window_features.m | Use 3-second windows and 0.5-second hop to extract 8 original plus 8 PSD band-power features |
| step04_train_linear_svm.m | Train a two-feature linear SVM, display boundary, test on **held-out subjects** and plot confusion matrix |
| step05_compare_power_ratio.m | Compare four feature choices with leave-one-training-subject-out CV; contrast absolute power with log high/low ratio |
| step06_compare_dnn.m | Compare SVM-2, SVM-8, SVM-16, DNN-8, and DNN-16 using identical subject-wise evaluation |

**Generated files stay local (ignored by Git):**

- <code>ME3310_USC_HAD_UpDown.mat</code>: <code>records</code>, <code>fs</code>, <code>classNames</code>. Each record is ONE continuous original trial.
- <code>ME3310_USC_HAD_WindowFeatures.mat</code>: numerical feature matrix <code>X</code>, labels <code>Y</code>, and subject/trial/record/time provenance.

## Labels, units and feature choice

- Activity 4: **Walking Upstairs**, label **0**
- Activity 5: **Walking Downstairs**, label **1**
- <code>records(i).acc</code>: N-by-3 acceleration, **g**
- <code>records(i).gyro</code>: N-by-3 angular velocity, **deg/s**
- <code>records(i).t</code>: seconds, from zero; sample rate <code>fs = 100 Hz</code>

The first eight features remain unchanged: acceleration-magnitude standard deviation; its 0.8–3 Hz PSD band power; X/Y/Z gyroscope RMS; and X/Y/Z signed gyroscope means. **Features 9-16 are additional acceleration-magnitude PSD band powers:** 0.3-0.8, 0.8-1.5, 1.5-3, 3-5, 5-8, 8-12, 12-20, and 20-40 Hz. The original 0.8-3 Hz feature overlaps two new bands, so some features are correlated; higher bands may contain sensor noise. All bands are exploratory, not guaranteed to help. **Run step02b to inspect the entire PSD before selecting a band.** It uses the training subjects only, weights subjects equally, and visualizes candidate bands; the resulting plots are exploratory, NOT final test accuracy.

Script 04 initially uses **two features** (acceleration standard deviation and band power) so a **straight-line decision boundary** can be visualized. Change <code>featureIdx = [1 2]</code> to test other features. Choosing features based on held-out test results would bias the evaluation; treat such exploration as preliminary and reserve an independent evaluation set.

## Experimental follow-up: power ratio

Run `step05_compare_power_ratio.m` after preparing the original MAT file. This script independently recomputes a consistent set of 3-second sliding-window features for STD alone, STD + low-band power, STD + high-band power, and STD + 10log10(high/low). It uses **the ten training subjects** for subject-wise cross-validation. An optional diagnostic reuses the previously reported four held-out subjects, which must not be described as a new independent test.

The log-ratio was motivated by **subject-level relative PSD differences**, but those plots alone do not establish that window-level log-ratio features discriminate the activities. Recheck band selection using truly new subjects or trials before making a generalization claim.

## Multi-subject DNN experiment

**After pulling the updated scripts, rerun step03** so the local feature MAT file has 16 columns. Then run `step06_compare_dnn.m`. It evaluates 5 configurations: SVM-2, SVM-8, SVM-16, DNN-8, and DNN-16, with the exact same subject-wise train/validation folds. Both DNN variants use a two-hidden-layer MLP fitted with MATLAB `fitcnet` (Statistics and Machine Learning Toolbox), not a raw-signal 1D CNN. Comparing 8 versus 16 features within the same model tests whether additional PSD bands help. The four previously seen subjects are a **secondary diagnostic**, not a fresh test.

## Important experimental limitations

- Overlapping windows from the same trial are **correlated**. They are not independent experimental repetitions.
- Script 04 reserves **entire subjects** for testing rather than mixing windows from the same person across training and test sets.
- The reported window accuracy is not a confidence interval. Examine variation by subject and acquire more independent trials.
- USC-HAD trials each contain **one** activity. The optional time-varying prediction plot does **not** demonstrate real Upstairs-to-Downstairs transitions.
- A model trained here does not automatically apply to a student's PocketLab device. Sampling rate, sensor orientation, units, wearing position and class definitions may differ.
- PSD removes sign and phase information. Whether it is useful for discriminating these specific motions is a question for the data.

## Requirements

- **MATLAB**: preparation and time plots.
- **Signal Processing Toolbox**: Welch PSD via <code>pwelch</code>.
- **Statistics and Machine Learning Toolbox**: <code>fitcsvm</code>, <code>confusionchart</code>.
- MATLAB **R2023b or newer** recommended for the repository's built-in <code>gitrepo</code>/<code>pull</code> update commands.
- The scripts were reviewed statically but **have not been run in a local MATLAB installation**.

Check locally:

~~~matlab
ver
which pwelch
which fitcsvm
~~~

## Engineering questions for ME 3310

1. Which measured variables separate Upstairs and Downstairs?
2. Does a PSD-derived feature improve on a simple time-domain feature?
3. Does a linear boundary work on unseen subjects or trials?
4. Why does a large number of overlapping windows not eliminate uncertainty?
5. How would you test the same classifier on a *continuous signal with motion transitions*?

## Source and permissions

Official dataset: [USC-SIPI Human Activity Dataset](https://sipi.usc.edu/had/).

Citation: Mi Zhang and Alexander A. Sawchuk, *USC-HAD: A Daily Activity Dataset for Ubiquitous Activity Recognition Using Wearable Sensors*, SAGAware Workshop (2012).

**No original sensor data are committed to this repository.** Download them from the official provider and check the redistribution conditions before sharing copies.
