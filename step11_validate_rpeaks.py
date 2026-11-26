"""Validates our R-peak detector (used for the WESAD and GalaxyPPG references)
on PPG-DaLiA, whose reference HR comes from the dataset authors.

Self-contained: the detector below is a copy of ppgs.data.r_peaks and
ppgs.data.window_hr_from_peaks. Usage, from the repository root:
    python step11_validate_rpeaks.py <folder that contains PPG_FieldStudy>
Writes results/rpeak_validation_dalia.csv and prints the agreement.
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

FS_ECG, WIN_S, SHIFT_S = 700, 8, 2


def _bandpass(x, fs, lo, hi, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def r_peaks(ecg, fs=FS_ECG):
    x = _bandpass(ecg - np.median(ecg), fs, 5, 20)
    env = np.convolve(np.gradient(x) ** 2, np.ones(int(0.15 * fs)) / int(0.15 * fs), "same")
    step, half = 2 * fs, 5 * fs
    centers = np.arange(0, len(env), step)
    local = np.array([np.percentile(env[max(0, c - half): c + half], 99) for c in centers])
    thr = 0.3 * np.interp(np.arange(len(env)), centers, local)
    cand, _ = find_peaks(env, distance=int(0.3 * fs))
    cand = cand[env[cand] > thr[cand]]
    sign = 1.0 if np.percentile(x, 99.9) >= -np.percentile(x, 0.1) else -1.0
    w = int(0.1 * fs)
    peaks = np.array([max(0, c - w) + np.argmax(sign * x[max(0, c - w): c + w]) for c in cand])
    return np.unique(peaks)


def window_hr_from_peaks(peaks, fs, n_win):
    t = peaks / fs
    rr, tm = np.diff(t), (t[1:] + t[:-1]) / 2
    med = np.array([np.median(rr[max(0, k - 5): k + 6]) for k in range(len(rr))])
    ok = (rr > 0.3) & (rr < 2.0) & (np.abs(rr - med) < 0.3 * med)
    hr = np.full(n_win, np.nan)
    for i in range(n_win):
        sel = (tm >= i * SHIFT_S) & (tm < i * SHIFT_S + WIN_S)
        if sel.sum() >= 4 and ok[sel].mean() >= 0.7:
            hr[i] = np.mean(60.0 / rr[sel & ok])
    return hr


import sys
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
rows = []
for sid in range(1, 16):
    path = ROOT / "PPG_FieldStudy" / f"S{sid}" / f"S{sid}.pkl"
    print(f"S{sid} ...", flush=True)
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    ecg = np.asarray(d["signal"]["chest"]["ECG"]).ravel().astype(float)
    ref = np.asarray(d["label"]).astype(float).ravel()
    del d
    ours = window_hr_from_peaks(r_peaks(ecg, FS_ECG), FS_ECG, len(ref))
    ok = ~np.isnan(ours)
    err = ours[ok] - ref[ok]
    rows.append({"subject": f"D{sid}", "windows": len(ref), "kept by QC (%)": 100 * ok.mean(),
                 "MAE (bpm)": np.abs(err).mean(), "bias (bpm)": err.mean(),
                 "within 5 bpm (%)": 100 * np.mean(np.abs(err) <= 5)})
    print(rows[-1], flush=True)
res = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
res.to_csv(Path("results") / "rpeak_validation_dalia.csv", index=False)
print(res.round(2).to_string(index=False))
print("\nMean over participants:\n", res.drop(columns="subject").mean().round(2).to_string())
