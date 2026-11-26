# Analysis plan: second external evaluation on PPG-Sport

Written and frozen, before any heart rate, error or coverage was computed on PPG-Sport.
Before writing it, the repository structure, the README, the participant table and the signals of one subject (S01, left wrist, stationary and running) were inspected to check the file format, the units and the signal quality.
Any deviation from this plan will be reported in the paper as a deviation, with its reason.

## Purpose

Evaluate, on new wearers and during sports with intense and non-periodic motion (running, badminton, table tennis, basketball), the calibrations already evaluated on GalaxyPPG. Nothing in the method is changed or tuned.

## Data

- PPG-Sport (Hu et al., Proc. ACM IMWUT 10(2), 2026, doi 10.1145/3810216), public repository LaserHu/PPG-Sport, all 30 subjects, six activities of 8 min each.
- Files as distributed (`dataset/`): wrist PPG `SxxW_act_ppg.npy` (1 channel), wrist IMU `SxxW_act_imu.npy` (3 axes, in g), ECG `Sxx_act_ecg.npy`, all at 64 Hz and synchronized by the authors.
- The repository does not name the wrist device. According to the dataset description (to be confirmed in the IMWUT paper), the wrist signals come from an Empatica EmbracePlus and the ECG from a Polar H10. The PPG is distributed already processed: its spectrum has almost no energy above 8 Hz.
- Reference: the ECG-derived HR provided in `cleaned_gt/` (one value per 8-s window shifted by 2 s, 237 windows per recording), as for PPG-DaLiA, where the reference distributed with the data was used.

## Wrist

- Main analysis: left wrist (chosen before any result; the participant table does not report handedness).
- Secondary analysis: right wrist, with exactly the same rules, reported separately. The two wrists are never pooled in one calibration evaluation.

## Preprocessing

1. The IMU (64 Hz) is downsampled to 32 Hz with an anti-aliasing polyphase filter (`scipy.signal.resample_poly(x, 1, 2)`) to match the E4 accelerometer rate used during development. Units are already in g and are not rescaled.
2. Each recording (one subject, one wrist, one activity) is processed independently by MAST, since the six activity files are separate sessions.
3. Windows of 8 s shifted by 2 s, aligned with the `cleaned_gt` windows (window k covers samples 128k to 128k + 511). Windows with a missing or non-finite reference are discarded.
4. MAST, the three descriptors and the window grid are unchanged.

## Calibration

The scale model and all thresholds are those obtained once from the 30 development participants (PPG-DaLiA and WESAD) and already applied to GalaxyPPG: window-level, person-level (level 93.0%), person-quantile (k = 28), and motion-stratified with the same motion-class edges. Main score: normalized. No threshold is recomputed on PPG-Sport.

## Outcomes (same order as in the paper)

1. Number of participants whose coverage reaches 90%.
2. Coverage of the worst participant.
3. Median interval width.
Also reported: mean coverage, mean width, MAE of MAST, coverage per activity, distribution of windows over the three motion classes compared with development, participant bootstrap (2000 resamples, thresholds fixed) for the differences between calibrations, and the comparison with one global threshold (Table III).

## Inclusion

All subjects with at least one valid window are included. Windows excluded by the rules above are counted and reported.

## What will not be done

No change of MAST settings, descriptors, scale model, thresholds, motion classes or criteria after seeing the PPG-Sport results. No subject is excluded on the basis of its coverage or error. No rescaling of the accelerometer to match the E4.

## Outcome, 24 November 2026

The plan was followed without deviation (step14_ppgsport_external.py). MAST did not transfer to this wristband (MAE 25.5 bpm left, 37.7 bpm right), and at most 2 of 30 participants reached the target with any calibration (Supplementary Section S10). After these results, PPG-Sport was used as an additional source of calibration persons in an exploratory cross-sensor experiment, tested on GalaxyPPG (step15_calibration_motion_rich.py, Section IV-E). This second use was not part of the plan and is reported as exploratory.
