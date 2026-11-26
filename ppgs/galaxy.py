"""GalaxyPPG (Park et al., 2025): independent test set, Empatica E4 + Polar H10.

Expected layout (Zenodo 10.5281/zenodo.14635823), one folder per participant:
    P02/E4/BVP.csv          wrist PPG, 64 Hz (timestamp, value)
    P02/E4/ACC.csv          wrist accelerometer, 32 Hz (timestamp, x, y, z)
    P02/PolarH10/ECG.csv    chest ECG, 130 Hz
    P02/PolarH10/ACC.csv    chest accelerometer, 200 Hz (clock alignment only)
    P02/Event.csv           start (ENTER) and end (EXIT) of each protocol phase

Each device has its own clock. For each file:
  1. timestamps are converted to ms and an integer-hour offset with
     Event.csv (time zone) is removed,
  2. timestamps often arrive in packets (Bluetooth): within each segment
     without interruption, sample times are rebuilt by a straight line
     fitted on the sample index,
  3. the chest clock is aligned on the wrist clock by cross-correlating
     the motion envelopes of the two accelerometers (+/- 60 s). This
     alignment uses neither the PPG nor the heart rate,
  4. signals are resampled on a common grid. Windows that touch an
     interruption of any stream are discarded.
The reference HR is computed from the ECG with the same rules as WESAD.
P01 has no Event.csv and is skipped.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import FS_ACC, FS_BVP, SHIFT_S, WIN_S, acc_in_g, r_peaks, window_hr_from_peaks

FS_ECG, FS_POLAR_ACC = 130, 200
GAP_MS = 1500
SYNC_BIN_MS, SYNC_MAX_S = 500, 60
SYNC_MIN_CORR, SYNC_MIN_SHARP = 0.4, 0.15

PHASES = {
    "baseline": "baseline", "tsst-prep": "TSST preparation", "tsst-speech": "TSST speech",
    "meditation-1": "meditation", "meditation-2": "meditation", "screen-reading": "screen reading",
    "ssst-prep": "SSST preparation", "ssst-sing": "SSST singing", "keyboard-typing": "keyboard typing",
    "mobile-typing": "mobile typing", "standing": "standing", "walking": "walking",
    "jogging": "jogging", "running": "running", "rest-1": "rest", "rest-2": "rest", "rest-3": "rest",
    "rest-4": "rest", "rest-5": "rest", "adaptation": "adaptation",
}


def list_galaxy(root):
    return [int(d.name[1:]) for d in sorted(Path(root).iterdir())
            if d.is_dir() and d.name[:1] == "P" and d.name[1:].isdigit()
            and any("event" in p.name.lower() for p in d.iterdir())]


def _read(path):
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _col(df, *names):
    low = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    raise KeyError(f"none of {names} in {list(df.columns)}")


def _to_ms(t):
    t = np.asarray(t, dtype=float)
    med = np.nanmedian(t)
    return t / 1000.0 if med > 1e14 else (t * 1000.0 if med < 1e11 else t)


def _hour_shift(t_ms, ref_ms):
    diff = np.nanmedian(ref_ms) - np.nanmedian(t_ms)
    return 0.0 if abs(diff) < 30 * 60 * 1000 else float(np.clip(np.round(diff / 3.6e6), -14, 14) * 3.6e6)


def _sample_times(ts, fs):
    """Sample times rebuilt by a line fit within each uninterrupted segment."""
    n = len(ts)
    cuts = np.flatnonzero(np.diff(ts) > GAP_MS) + 1
    bounds = list(zip(np.r_[0, cuts], np.r_[cuts, n]))
    t = np.empty(n)
    for a, b in bounds:
        idx = np.arange(a, b)
        slope, icpt = 1000.0 / fs, ts[a] - 1000.0 / fs * a
        if b - a >= 10:
            s, c = np.polyfit(idx, ts[a:b], 1)
            if 0.5 * 1000 / fs < s < 2.0 * 1000 / fs:
                slope, icpt = s, c
            else:
                icpt = np.median(ts[a:b] - slope * idx)
        t[a:b] = icpt + slope * idx
    return t, bounds


def _on_grid(t, values, bounds, grid):
    values = np.asarray(values, dtype=float)
    out = np.full((len(grid),) + values.shape[1:], np.nan)
    ok = np.zeros(len(grid), dtype=bool)
    for a, b in bounds:
        if b - a < 2:
            continue
        m = (grid >= t[a]) & (grid <= t[b - 1])
        if values.ndim == 1:
            out[m] = np.interp(grid[m], t[a:b], values[a:b])
        else:
            for k in range(values.shape[1]):
                out[m, k] = np.interp(grid[m], t[a:b], values[a:b, k])
        ok |= m
    return out, ok


def _envelope(t, xyz, T0, n_bins):
    m = np.linalg.norm(np.asarray(xyz, dtype=float), axis=1)
    k = np.floor((t - T0) / SYNC_BIN_MS).astype(int)
    ok = (k >= 0) & (k < n_bins) & np.isfinite(m)
    g = pd.Series(m[ok]).groupby(k[ok])
    sd, cnt = g.std(), g.size()
    env = np.full(n_bins, np.nan)
    good = cnt[cnt >= 3].index
    env[good] = sd[good].to_numpy()
    eps = 0.01 * np.nanmedian(env[env > 0]) if np.any(env > 0) else 1e-6
    return np.log(env + eps)


def _sync_offset(t_w, acc_w, t_c, acc_c):
    """Chest clock offset (ms) from the cross-correlation of motion envelopes."""
    T0 = max(t_w[0], t_c[0])
    n = int((min(t_w[-1], t_c[-1]) - T0) // SYNC_BIN_MS)
    if n < 600:
        return 0.0, np.nan, np.nan
    e, p = _envelope(t_w, acc_w, T0, n), _envelope(t_c, acc_c, T0, n)
    lags = np.arange(-int(SYNC_MAX_S * 1000 / SYNC_BIN_MS), int(SYNC_MAX_S * 1000 / SYNC_BIN_MS) + 1)
    corr = np.full(len(lags), np.nan)
    for j, L in enumerate(lags):
        a, b = (e[: n - L], p[L:]) if L >= 0 else (e[-L:], p[: n + L])
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 120:
            corr[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    if not np.isfinite(corr).any():
        return 0.0, np.nan, np.nan
    j = int(np.nanargmax(corr))
    frac = 0.0
    if 0 < j < len(lags) - 1 and np.isfinite(corr[j - 1]) and np.isfinite(corr[j + 1]):
        den = corr[j - 1] - 2 * corr[j] + corr[j + 1]
        frac = float(np.clip(0.5 * (corr[j - 1] - corr[j + 1]) / den, -0.5, 0.5)) if den != 0 else 0.0
    return float((lags[j] + frac) * SYNC_BIN_MS), float(corr[j]), float(corr[j] - np.nanmedian(corr))


def load_galaxy(root, sid):
    folder = Path(root) / f"P{sid:02d}"
    ev = _read(next(p for p in folder.iterdir() if p.suffix.lower() == ".csv" and "event" in p.name.lower()))
    ev_t = _to_ms(ev[_col(ev, "timestamp", "time")].to_numpy())
    ev_s = ev[_col(ev, "session", "event")].astype(str).str.strip().str.lower().to_numpy()
    ev_st = ev[_col(ev, "status", "type")].astype(str).str.strip().str.upper().to_numpy()

    def device(sub, name, fs, value_cols, time_cols=("timestamp",)):
        df = _read(folder / sub / name)
        tcol = _col(df, *time_cols)
        df = df.dropna(subset=[tcol]).sort_values(tcol, kind="mergesort")
        t = _to_ms(df[tcol].to_numpy())
        t = t + _hour_shift(t, ev_t)
        vals = np.column_stack([pd.to_numeric(df[_col(df, *c)], errors="coerce").to_numpy() for c in value_cols])
        vals = pd.DataFrame(vals).interpolate(limit_direction="both").to_numpy()
        t_fit, bounds = _sample_times(t, fs)
        return t_fit, bounds, (vals[:, 0] if len(value_cols) == 1 else vals)

    tb, bb, bvp = device("E4", "BVP.csv", FS_BVP, [("value", "bvp")])
    ta, ba, acc = device("E4", "ACC.csv", FS_ACC, [("x",), ("y",), ("z",)])
    te, be, ecg = device("PolarH10", "ECG.csv", FS_ECG, [("ecg",)], ("phoneTimestamp", "timestamp"))
    acc = acc_in_g(acc)

    shift, info = 0.0, (np.nan, np.nan, False)
    try:
        tp, _, accp = device("PolarH10", "ACC.csv", FS_POLAR_ACC, [("X",), ("Y",), ("Z",)],
                             ("phoneTimestamp", "timestamp"))
        off, c, s = _sync_offset(ta, acc, tp, accp)
        applied = bool(np.isfinite(c) and c >= SYNC_MIN_CORR and s >= SYNC_MIN_SHARP)
        shift, info = (off if applied else 0.0), (off / 1000, c, applied)
    except (FileNotFoundError, KeyError):
        pass
    te = te - shift

    t0, t1 = max(tb[0], ta[0], te[0]), min(tb[-1], ta[-1], te[-1])
    dur = (t1 - t0) / 1000.0
    bvp_g, ok_b = _on_grid(tb, bvp, bb, t0 + np.arange(int(dur * FS_BVP)) * 1000.0 / FS_BVP)
    acc_g, ok_a = _on_grid(ta, acc, ba, t0 + np.arange(int(dur * FS_ACC)) * 1000.0 / FS_ACC)
    n_win = int((dur - WIN_S) // SHIFT_S) + 1

    peak_s = []
    for a, b in be:
        if b - a >= 10 * FS_ECG:
            peak_s.append((te[a:b][r_peaks(ecg[a:b], FS_ECG)] - t0) / 1000.0)
    peak_s = np.sort(np.concatenate(peak_s)) if peak_s else np.array([])
    hr = window_hr_from_peaks(peak_s, 1, n_win)

    starts = np.arange(n_win) * SHIFT_S
    ok = np.zeros(n_win, dtype=bool)
    for a, b in be:
        ok |= (starts >= (te[a] - t0) / 1000.0) & (starts + WIN_S <= (te[b - 1] - t0) / 1000.0)
    for i in np.flatnonzero(ok):
        ok[i] = ok_b[i * SHIFT_S * FS_BVP: i * SHIFT_S * FS_BVP + WIN_S * FS_BVP].all() and \
            ok_a[i * SHIFT_S * FS_ACC: i * SHIFT_S * FS_ACC + WIN_S * FS_ACC].all()
    hr[~ok] = np.nan

    # protocol phase at the window centre
    centre_ms = t0 + (starts + WIN_S / 2) * 1000.0
    cond = np.array(["transition"] * n_win, dtype=object)
    order = np.argsort(ev_t, kind="mergesort")
    ev_t, ev_s, ev_st = ev_t[order], ev_s[order], ev_st[order]
    for i in np.flatnonzero(ev_st == "ENTER"):
        later = np.flatnonzero((ev_s == ev_s[i]) & (ev_st == "EXIT") & (ev_t >= ev_t[i]))
        if len(later) and ev_s[i] in PHASES:
            cond[(centre_ms >= ev_t[i]) & (centre_ms < ev_t[later[0]])] = PHASES[ev_s[i]]

    return {"dataset": "GalaxyPPG", "subject": f"G{sid}", "bvp": np.nan_to_num(bvp_g),
            "acc": np.nan_to_num(acc_g), "hr_ref": hr, "condition": cond.astype(str),
            "sync": info}
