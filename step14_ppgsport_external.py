"""Step 14: MAST and the development calibrations applied directly to PPG-Sport (plan: ANALYSIS_PLAN_PPG_SPORT.md).

Usage: python step14_ppgsport_external.py <folder with the six activity packs> [L|R]
Packs: <activity>_<wrist>.npz (or <activity>.npz) with SxxW_ppg, SxxW_imu (64 Hz, g) and Sxx_hr (cleaned_gt).
Nothing is fitted on PPG-Sport: MAST, the scale model and every threshold come
from the 30 development participants (results/windows.csv).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from ppgs.estimator import process_record
from ppgs.conformal import METHODS, N_FOLDS, HR_RANGE, ALPHA, weighted_quantile, person_folds

ACTS = ["stationary", "walking", "running", "badminton", "tabletennis", "basketball"]
F3 = ["snr_db", "motion", "disagree"]
DELTA = 0.10
root = Path(sys.argv[1]); wrist = sys.argv[2] if len(sys.argv) > 2 else "L"
out = Path("results")


def ceil_int(x):
    return int(np.ceil(round(x, 9)))


def q_window(s, subj):
    n = len(s); lev = (1 - ALPHA) * (1 + 1 / n)
    return np.inf if lev > 1 else weighted_quantile(s, np.ones(n), lev)


def q_person(s, subj):
    _, inv, cnt = np.unique(subj, return_inverse=True, return_counts=True)
    lev = (1 - ALPHA) * (1 + 1 / len(cnt))
    return np.inf if lev > 1 else weighted_quantile(s, 1.0 / cnt[inv], lev)


def q_quantile(s, subj):
    r = np.sort([np.sort(s[subj == p])[ceil_int((1 - ALPHA) * (subj == p).sum()) - 1] for p in pd.unique(subj)])
    k = ceil_int((1 - DELTA) * (len(r) + 1))
    return np.inf if k > len(r) else r[k - 1]


# 1. windows of PPG-Sport (each activity file processed on its own), or a windows file computed beforehand
if root.suffix == ".csv":
    test = pd.read_csv(root)
else:
    parts = []
    for act in ACTS:
        f_ = root / f"{act}_{wrist}.npz"
        d = np.load(f_ if f_.exists() else root / f"{act}.npz")
        for i in range(1, 31):
            s = f"S{i:02d}"
            if f"{s}{wrist}_ppg" not in d:
                continue
            ppg = np.asarray(d[f"{s}{wrist}_ppg"], dtype=float)
            if ppg.ndim > 1:
                ppg = ppg[:, 0] if ppg.shape[0] > ppg.shape[1] else ppg[0]
            imu = np.asarray(d[f"{s}{wrist}_imu"], dtype=float)
            if imu.shape[0] < imu.shape[1]:
                imu = imu.T
            acc = resample_poly(imu, 1, 2, axis=0)
            rec = {"bvp": ppg, "acc": acc,
                   "hr_ref": d[f"{s}_hr"].astype(float), "subject": s, "dataset": "PPG-Sport"}
            rec["condition"] = np.array([act] * len(rec["hr_ref"]))
            parts.append(process_record(rec))
        print(act, "done", flush=True)
    test = pd.concat(parts, ignore_index=True)
    test.to_csv(out / f"windows_ppgsport_{wrist}.csv", index=False)

# 2. scale model and thresholds from the development data (as for GalaxyPPG)
dev = pd.read_csv(out / "windows.csv")
X, e = dev[F3].to_numpy(), (dev.hr_ref - dev.hr_hat).to_numpy()
subj = dev.subject.to_numpy()
f = person_folds(subj, dev.dataset.to_numpy())
oof = np.empty(len(e))
for k in range(N_FOLDS):
    oof[f == k] = METHODS["normalized"]().fit(X[f != k], e[f != k]).score(X[f == k], e[f == k])
model = METHODS["normalized"]().fit(X, e)
sig = model.sigma(test[F3].to_numpy())
thr = {"window-level": q_window(oof, subj), "person-level": q_person(oof, subj),
       "person-quantile": q_quantile(oof, subj)}
edges = np.quantile(dev.motion, [1 / 3, 2 / 3])
cls_dev, cls_te = np.digitize(dev.motion, edges), np.digitize(test.motion, edges)
q_strat = np.full(len(test), np.nan)
for c in np.unique(cls_te):
    q_strat[cls_te == c] = q_person(oof[cls_dev == c], subj[cls_dev == c])
print("thresholds", {k: round(v, 3) for k, v in thr.items()}, "edges", edges.round(3))

# 3. outcomes
rows, act_rows = [], []
for name, q in list(thr.items()) + [("motion-stratified", q_strat)]:
    lo = np.maximum(test.hr_hat - q * sig, HR_RANGE[0]); hi = np.minimum(test.hr_hat + q * sig, HR_RANGE[1])
    cov = (test.hr_ref >= lo) & (test.hr_ref <= hi); w = hi - lo
    per = cov.groupby(test.subject).mean()
    rows.append({"wrist": wrist, "calibration": name, "at target": int((per >= 1 - ALPHA).sum()), "persons": len(per),
                 "worst": per.min(), "mean cov": per.mean(), "median width": w.median(), "mean width": w.mean()})
    for a, g in cov.groupby(test.condition):
        act_rows.append({"wrist": wrist, "calibration": name, "activity": a, "coverage": g.mean(),
                         "median width": w[g.index].median(), "windows": len(g)})
tab = pd.DataFrame(rows); tab.to_csv(out / f"summary_ppgsport_{wrist}.csv", index=False)
pd.DataFrame(act_rows).to_csv(out / f"summary_ppgsport_{wrist}_activities.csv", index=False)

mae = (test.hr_hat - test.hr_ref).abs()
per_mae = mae.groupby(test.subject).mean()
print(f"\nPPG-Sport ({wrist}): {test.subject.nunique()} participants, {len(test)} windows, "
      f"MAE {per_mae.mean():.2f} +/- {per_mae.std():.2f} bpm")
print("MAE by activity:", mae.groupby(test.condition).mean().round(1).to_dict())
print("motion classes, development:", np.bincount(cls_dev, minlength=3) / len(cls_dev))
print("motion classes, PPG-Sport:  ", np.bincount(cls_te, minlength=3) / len(cls_te))
print(pd.crosstab(test.condition, cls_te, normalize="index").round(2))
pd.set_option("display.width", 250)
print(tab.round(3).to_string(index=False))
print(pd.DataFrame(act_rows).pivot(index="activity", columns="calibration", values="coverage").round(3))
