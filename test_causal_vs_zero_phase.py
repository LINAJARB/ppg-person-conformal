"""Separate check, not part of the main pipeline: causal versus zero-phase filtering in MAST.

Both variants use a 4th-order Butterworth band-pass filter, 0.5-4 Hz:
  zero-phase: scipy.signal.filtfilt (forward and backward pass, amplitude |H|^2, no delay), as in the paper
  causal:     scipy.signal.sosfilt (single forward pass, amplitude |H|, causal)
The filter is replaced for the PPG and the accelerometer, hence for the spectral tracker, the motion
mask and the beat detection (disagreement descriptor). The ECG reference of WESAD is unchanged.
Only the MAE and the mean disagreement are compared. The scale model, the thresholds and the coverage
were NOT recomputed with the causal variant.

Usage: python test_causal_vs_zero_phase.py <compact folder>
Output: results/causal_check.csv (one row per participant and variant)
"""
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
import ppgs.estimator as E
from ppgs.data import load_dalia, load_wesad, DALIA_IDS, WESAD_IDS

folder = sys.argv[1] if len(sys.argv) > 1 else "compact"


def causal_bandpass(x, fs, lo=0.5, hi=4.0, order=4):
    sos = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band", output="sos")
    return sosfilt(sos, x, axis=0)


rows = []
zero_phase = E.bandpass
for mode, bp in [("zero-phase", zero_phase), ("causal", causal_bandpass)]:
    E.bandpass = bp
    for loader, ids in [(load_dalia, DALIA_IDS), (load_wesad, WESAD_IDS)]:
        for sid in ids:
            df = E.process_record(loader(folder, sid))
            rows.append({"variant": mode, "subject": df.subject[0], "dataset": df.dataset[0],
                         "mae": (df.hr_hat - df.hr_ref).abs().mean(), "mean_disagree": df.disagree.mean(),
                         "mean_snr_db": df.snr_db.mean(), "windows": len(df)})
    print(mode, "done", flush=True)
E.bandpass = zero_phase
r = pd.DataFrame(rows)
r.to_csv("results/causal_check.csv", index=False)
print(r.groupby(["variant", "dataset"])[["mae", "mean_disagree", "mean_snr_db"]].mean().round(2))
