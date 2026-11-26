"""MAST (motion-attenuated spectral tracker). Heart-rate estimator and the three signal-quality descriptors.

Estimator (no learning, online tracking), for each 8 s window shifted by 2 s.
The band-pass filter is zero phase (filtfilt over the recording), so the
estimator as implemented is not strictly causal:
  1. band-pass 0.5-4 Hz of the PPG and of the three accelerometer axes,
  2. zero-padded spectra, max-pooled on a 1 bpm grid from 40 to 200 bpm,
  3. motion attenuation: S(f) = P(f) (1 - m A(f))^2, where A is the
     accelerometer spectrum (normalized to 1, widened by +/- 2 bpm) and
     m in [0, 1] is the motion level of the window,
  4. tracking by dynamic programming (online Viterbi): the HR follows a
     Gaussian random walk (sd 3 bpm per 2 s step), with a small
     probability of jumping anywhere so that the tracker can recover.
     The estimate at time t is the most likely state given the windows
     up to t only (no look-ahead).

Descriptors, each tied to one cause of error:
  snr_db    spectral SNR around the retained peak and its harmonic
            (is the cardiac peak visible?)
  motion    log10 of the accelerometer-magnitude standard deviation, in g
            (is the wrist moving?)
  disagree  |HR from the spectrum - HR from beat detection|, in bpm
            (do two independent estimates agree?)
"""

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, filtfilt, find_peaks

from .data import FS_ACC, FS_BVP, SHIFT_S, WIN_S

NFFT = 8192                       # spectral resolution 0.47 bpm before pooling
GRID = np.arange(40.0, 200.5, 1.0)  # HR grid, bpm
MOTION_REF = 0.05                 # accelerometer std (g) for full attenuation
MASK_WIDTH = 2                    # widening of the motion mask, bpm
STEP_SD = 3.0                     # random-walk sd of the HR between windows, bpm
JUMP = 12.0                       # log-penalty of a jump outside the random walk
MAX_STEP = 20                     # largest step considered by the random walk, bpm
OBS_POWER = 0.5                   # weight of the spectrum in the log-likelihood

FEATURES = ["snr_db", "motion", "disagree"]


def bandpass(x, fs, lo=0.5, hi=4.0, order=4):
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x, axis=0)


def power_spectrum(x, nfft=NFFT):
    """Hann-windowed power spectrum (summed over columns for 2-D input)."""
    w = np.hanning(len(x))
    x = x - x.mean(axis=0)
    x = x * (w[:, None] if x.ndim > 1 else w)
    p = np.abs(np.fft.rfft(x, n=nfft, axis=0)) ** 2
    return p.sum(axis=1) if p.ndim > 1 else p


def pool_on_grid(p, f):
    """Max of the spectrum within +/- 0.5 bpm of each grid point."""
    bpm = 60 * f
    idx = np.searchsorted(bpm, GRID - 0.5), np.searchsorted(bpm, GRID + 0.5)
    return np.array([p[a:b].max() if b > a else 0.0 for a, b in zip(*idx)])


def snr_db(p, f, f0):
    """Power within 0.1 Hz of f0 and 2 f0, over the rest of 0.5-4 Hz."""
    band = (f >= 0.5) & (f <= 4.0)
    sig = band & ((np.abs(f - f0) <= 0.1) | (np.abs(f - 2 * f0) <= 0.1))
    return 10 * np.log10(p[sig].sum() / max(p[band & ~sig].sum(), 1e-12))


def hr_from_beats(x, fs):
    """HR from the median inter-beat interval of the filtered PPG."""
    pk, _ = find_peaks(x, distance=int(0.3 * fs), prominence=0.3 * np.std(x))
    if len(pk) < 3:
        return np.nan
    return 60.0 * fs / np.median(np.diff(pk))


def online_viterbi(S):
    """Causal dynamic-programming tracking of the HR on GRID.

    S has one row per window (attenuated spectrum on GRID). Returns the
    most likely HR at each window given the past and present windows."""
    G = len(GRID)
    steps = np.arange(-MAX_STEP, MAX_STEP + 1)
    src = np.arange(G)[:, None] + steps[None, :]
    valid = (src >= 0) & (src < G)
    src = np.clip(src, 0, G - 1)
    log_step = -0.5 * (steps / STEP_SD) ** 2
    obs = OBS_POWER * np.log(S / np.maximum(S.sum(axis=1, keepdims=True), 1e-12) + 1e-6)

    delta = obs[0].copy()
    out = np.empty(len(S))
    out[0] = GRID[np.argmax(delta)]
    for t in range(1, len(S)):
        local = np.where(valid, delta[src] + log_step, -np.inf).max(axis=1)
        delta = np.maximum(local, delta.max() - JUMP) + obs[t]
        delta -= delta.max()
        out[t] = GRID[np.argmax(delta)]
    return out


def process_record(rec):
    """Estimated HR and descriptors for every window with a valid reference."""
    bvp = bandpass(rec["bvp"], FS_BVP)
    acc = bandpass(rec["acc"], FS_ACC)
    acc_mag = np.linalg.norm(rec["acc"], axis=1)
    f_ppg = np.fft.rfftfreq(NFFT, 1 / FS_BVP)
    f_acc = np.fft.rfftfreq(NFFT, 1 / FS_ACC)
    n_win = min(len(rec["hr_ref"]),
                (len(bvp) - WIN_S * FS_BVP) // (SHIFT_S * FS_BVP) + 1,
                (len(acc) - WIN_S * FS_ACC) // (SHIFT_S * FS_ACC) + 1)

    spectra, S, motion_std, hr_beats = [], np.empty((n_win, len(GRID))), np.empty(n_win), np.empty(n_win)
    for i in range(n_win):
        x = bvp[i * SHIFT_S * FS_BVP: i * SHIFT_S * FS_BVP + WIN_S * FS_BVP]
        a = acc[i * SHIFT_S * FS_ACC: i * SHIFT_S * FS_ACC + WIN_S * FS_ACC]
        m = acc_mag[i * SHIFT_S * FS_ACC: i * SHIFT_S * FS_ACC + WIN_S * FS_ACC]
        p = power_spectrum(x)
        pa = maximum_filter1d(pool_on_grid(power_spectrum(a), f_acc), 2 * MASK_WIDTH + 1)
        motion_std[i] = np.std(m)
        level = min(1.0, motion_std[i] / MOTION_REF)
        S[i] = pool_on_grid(p, f_ppg) * np.clip(1 - level * pa / max(pa.max(), 1e-12), 0, 1) ** 2
        spectra.append(p)
        hr_beats[i] = hr_from_beats(x, FS_BVP)

    hr_hat = online_viterbi(S)
    df = pd.DataFrame({
        "window": np.arange(n_win),
        "hr_hat": hr_hat,
        "snr_db": [snr_db(p, f_ppg, h / 60) for p, h in zip(spectra, hr_hat)],
        "motion": np.log10(motion_std + 1e-3),
        "disagree": np.where(np.isfinite(hr_beats), np.minimum(np.abs(hr_hat - hr_beats), 60.0), 60.0),
    })
    df["hr_ref"] = rec["hr_ref"][:n_win]
    df["condition"] = rec["condition"][:n_win]
    df["subject"] = rec["subject"]
    df["dataset"] = rec["dataset"]
    return df[np.isfinite(df.hr_ref)].reset_index(drop=True)
