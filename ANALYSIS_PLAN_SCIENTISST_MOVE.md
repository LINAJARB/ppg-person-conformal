# Analysis plan: second external evaluation on ScientISST MOVE

Written and frozen, before any ScientISST MOVE file was downloaded or inspected.
Any deviation from this plan will be reported in the paper as a deviation, with its reason.

## Purpose

Evaluate, on new wearers and in a protocol with outdoor running, the calibrations already evaluated on GalaxyPPG. Nothing in the method is changed or tuned.

## Data

- ScientISST MOVE v1.0.1 (PhysioNet, CC BY 4.0), all 17 subjects.
- Estimator inputs: Empatica E4 `ppg-wrist` (64 Hz) and `acc-wrist-x/y/z` (32 Hz) from `empatica.edf`.
- Reference: chest ECG `ecg-gel` from `scientisst_chest.edf` (500 Hz). `ecg-dry` is not used.
- Not used: the ScientISST forearm device, the finger PPG, EDA, EMG, temperature.

## Preprocessing (same rules as GalaxyPPG)

1. Accelerometer converted to g with the same rule as for the other datasets (`acc_in_g`).
2. The two devices share a start mark. The chest clock is nevertheless checked against the wrist clock by cross-correlating the motion envelopes of the chest and wrist accelerometers (`acc-chest-x/y/z` and `acc-wrist-x/y/z`), with the GalaxyPPG settings: bins of 500 ms, lags of +/- 60 s, offset applied only if the correlation is at least 0.4 and its sharpness at least 0.15. This alignment uses neither the PPG nor the HR.
3. Windows of 8 s shifted by 2 s on the span common to the two devices.
4. Reference HR from the chest ECG with the R-peak detector and the window quality control used for WESAD and GalaxyPPG (sampling rate 500 Hz). Windows without a valid reference are discarded.
5. MAST, the three descriptors and the window grid are unchanged.

## Activity labels

From the EDF annotations: `baseline`, `lift` (including `lift-1`, `lift-2`), `greetings`, `gesticulate`, `jumps`, `walk_before` (including `walk_before_downstairs`), `run` (including `sprint`), `walk_after`. Windows outside any annotated activity are labelled `other`.

## Calibration

The scale model and all thresholds are those obtained once from the 30 development participants (PPG-DaLiA and WESAD) and already applied to GalaxyPPG: window-level, person-level (level 93.0%), person-quantile (k = 28), and motion-stratified with the same motion-class edges. Main score: normalized. No threshold is recomputed on ScientISST MOVE.

## Outcomes (same order as in the paper)

1. Number of participants whose coverage reaches 90%.
2. Coverage of the worst participant.
3. Median interval width.
Also reported: mean coverage, mean width, MAE of MAST, coverage per activity, participant bootstrap (2000 resamples, thresholds fixed) for the differences between calibrations, and the comparison with one global threshold (Table III).

## Inclusion

All subjects with at least one valid window are included. Subjects or windows excluded by the rules above are counted and reported. A subject with fewer than 50 valid windows is still included, and the number is reported.

## What will not be done

No change of MAST settings, descriptors, scale model, thresholds, motion classes or criteria after seeing the ScientISST MOVE results. No subject is excluded on the basis of its coverage or error.

## Data check (before any heart rate or coverage was computed)

Reading the raw 16-bit samples of the published EDF files shows that the wrist PPG (`ppg:wrist`) is clipped at the digital limits declared in the header. In all 15 subjects with both files (93JD has no `empatica.edf`, and LAS2 could not be transferred), 95% to 99% of its samples sit at one of the two extreme values, so the stored PPG is close to a square wave. The chest ECG (`ecg:gel`) is not clipped. Its flat segments are zero-filled interruptions, and its QRS complexes are intact. With the wrist PPG in this form, MAST and its descriptors cannot be applied as planned. The evaluation is therefore suspended. No estimate, error or coverage was computed on this dataset.
