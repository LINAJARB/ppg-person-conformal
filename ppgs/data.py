"""Data loading for PPG-DaLiA and WESAD (compact .npz copies).

Both datasets use the Empatica E4 wristband: PPG (BVP) at 64 Hz and a
3-axis accelerometer at 32 Hz. The reference heart rate (HR) is defined on
8 s windows shifted by 2 s, as in PPG-DaLiA.

PPG-DaLiA provides the reference HR directly (from the chest ECG).
WESAD only provides the chest ECG (700 Hz), so the reference is computed
here with the same convention: R peaks, instantaneous HR = 60 / RR, mean
over each 8 s window.
"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS_BVP, FS_ACC, FS_ECG = 64, 32, 700
WIN_S, SHIFT_S = 8, 2

DALIA_ACTIVITIES = {0: "transition", 1: "sitting", 2: "stairs", 3: "table soccer",
                    4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}
WESAD_CONDITIONS = {0: "transition", 1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}


def acc_in_g(acc):
    """E4 accelerometer in g (raw E4 units are 1/64 g; gravity norm = 1 g)."""
    acc = np.asarray(acc, dtype=float)
    if np.median(np.linalg.norm(acc, axis=1)) > 8:
        acc = acc / 64.0
    return acc


def majority(labels):
    vals, counts = np.unique(labels, return_counts=True)
    return int(vals[np.argmax(counts)])


def load_dalia(folder, sid):
    d = np.load(Path(folder) / f"dalia_S{sid}.npz")
    act = d["activity"]  # 4 Hz
    hr = d["hr_ref"].astype(float)
    labels = [DALIA_ACTIVITIES[majority(act[4 * SHIFT_S * i: 4 * SHIFT_S * i + 4 * WIN_S])]
              for i in range(len(hr))]
    return {"dataset": "DaLiA", "subject": f"D{sid}", "bvp": d["bvp"].astype(float),
            "acc": acc_in_g(d["acc"]), "hr_ref": hr, "condition": np.array(labels)}


def _bandpass(x, fs, lo, hi, order=3):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)


def r_peaks(ecg, fs=FS_ECG):
    """R peaks with a simple Pan-Tompkins variant (polarity detected)."""
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
    """Mean of 60 / RR in each 8 s window, NaN when too few valid RR."""
    t = peaks / fs
    rr, tm = np.diff(t), (t[1:] + t[:-1]) / 2
    # RR valid if physiological and within 30 % of the local median (11 beats)
    med = np.array([np.median(rr[max(0, k - 5): k + 6]) for k in range(len(rr))])
    ok = (rr > 0.3) & (rr < 2.0) & (np.abs(rr - med) < 0.3 * med)
    hr = np.full(n_win, np.nan)
    for i in range(n_win):
        sel = (tm >= i * SHIFT_S) & (tm < i * SHIFT_S + WIN_S)
        if sel.sum() >= 4 and ok[sel].mean() >= 0.7:
            hr[i] = np.mean(60.0 / rr[sel & ok])
    return hr


def load_wesad(folder, sid):
    d = np.load(Path(folder) / f"wesad_S{sid}.npz")
    bvp, acc, ecg, lab = d["bvp"].astype(float), acc_in_g(d["acc"]), d["ecg"].astype(float), d["label"]
    duration = min(len(bvp) / FS_BVP, len(acc) / FS_ACC, len(ecg) / FS_ECG)
    n_win = int((duration - WIN_S) // SHIFT_S) + 1
    hr = window_hr_from_peaks(r_peaks(ecg), FS_ECG, n_win)
    labels = [WESAD_CONDITIONS.get(majority(lab[4 * SHIFT_S * i: 4 * SHIFT_S * i + 4 * WIN_S]), "transition")
              for i in range(n_win)]
    return {"dataset": "WESAD", "subject": f"W{sid}", "bvp": bvp, "acc": acc,
            "hr_ref": hr, "condition": np.array(labels)}


DALIA_IDS = list(range(1, 16))
WESAD_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
