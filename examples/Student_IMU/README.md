# Student IMU measurements (ME 3310)

This folder provides a local MAT-file inspection and standardization layer for student measurements; it is NOT a new classifier. It was developed against two sample files named trial_T01.mat and trial_T02.mat (not committed to GitHub).

## Run

From the repository root, use MATLAB:

    repo = gitrepo;
    pull(repo);
    run("examples/Student_IMU/step01_inspect_student_mat.m")

Select one or more local MAT files in the dialog. The script checks timestamps, sampling frequency, number of samples, available sensor channels and optional event labels. It plots acceleration and/or gyro signals, including acceleration-magnitude PSD when Signal Processing Toolbox is available.

The reusable loader load_student_trial.m returns:
- trial.t: time in seconds, starting at zero
- trial.fs: rate estimated from timestamps
- trial.acc: Nx3 acceleration when present, otherwise empty
- trial.gyro: Nx3 angular velocity when present, otherwise empty
- trial.label: optional meta.EventLabel; never inferred from filename
- trial.meta: original measurement metadata
- trial.units: optional declared physical units (not automatically converted)

## Two actual student example recordings

- T01: 104 samples at 20 Hz, t=0..5.15 s
- T02: 100 samples at 20 Hz, t=0..4.95 s
- Both contain Nx3 accel, Nx1 t, meta, but NO gyro
- Measured magnitudes near one raw unit suggest but do not independently prove accelerometer unit g. Verify meta.Units.accel in MATLAB.

For 20 Hz acquisition, Nyquist is 10 Hz. The previous USC-HAD 8-12 Hz, 12-20 Hz and 20-40 Hz bands cannot be reused. A 2-second Welch segment also provides only about 0.5 Hz native resolution. Because no gyro was measured, never replace absent angular-rate channels with fabricated zeros.

T02 includes a magnitude peak of approximately 7.94 raw acceleration units at 3.10 s, largely from a Y-axis excursion. Verify whether this is real impact or an artifact. Both sample files report one invalid driver packet; the logs do not prove that the peak was caused by it.

## Before classification

Collect multiple independent, labeled trials per motion, confirm acceleration units, placement/orientation and sampling, select appropriate accel-only features, and split by trial or by subject before evaluation. Overlapping windows are NOT independent experimental replicates.

## Privacy

Do NOT add student raw data, Bluetooth addresses, raw sensor packets or other identifying device metadata to this repository. The local MAT/CSV files are gitignored in this folder. Only code is committed.

The MATLAB scripts have been reviewed structurally but not executed in a MATLAB installation.
