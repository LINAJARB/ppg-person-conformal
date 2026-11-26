"""Quantile crossing and empty intervals in CQR.

Replays the random subsampling of step02_development_calibration.py (DaLiA, WESAD, both, in this
order, in one process) and of step03_galaxyppg_external.py (GalaxyPPG, fresh process), so that
the final quantile models are the ones that produced the reported intervals.
Fold models only advance the random generator, except for the first held-out
subject of PPG-DaLiA, which is fully recomputed to check the replay against
results/intervals_DaLiA_cqr.csv.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from ppgs.conformal import (ALPHA, CQR_MAX_FIT, N_FOLDS, HR_RANGE, FEATURES,
                            person_folds, calibration_quantile)

rng = np.random.default_rng(0)   # same as the default argument of CQR.fit


def fit(X, e):
    idx = rng.choice(len(X), min(len(X), CQR_MAX_FIT), replace=False)
    lo = QuantileRegressor(quantile=ALPHA / 2, alpha=0.0, solver="highs").fit(X[idx], e[idx])
    hi = QuantileRegressor(quantile=1 - ALPHA / 2, alpha=0.0, solver="highs").fit(X[idx], e[idx])
    return lo, hi


def skip(n):
    rng.choice(n, min(n, CQR_MAX_FIT), replace=False)


def replay_loso(df, name, check_first=False):
    X, e = df[FEATURES].to_numpy(), (df.hr_ref - df.hr_hat).to_numpy()
    subj, ds = df.subject.to_numpy(), df.dataset.to_numpy()
    cross, n = 0, 0
    for i, t in enumerate(np.unique(subj)):
        te, tr = subj == t, subj != t
        folds = person_folds(subj[tr], ds[tr])
        Xtr, etr = X[tr], e[tr]
        full = check_first and i == 0
        oof = np.empty(tr.sum())
        for k in range(N_FOLDS):
            f = folds == k
            if full:
                lo, hi = fit(Xtr[~f], etr[~f])
                oof[f] = np.maximum(lo.predict(Xtr[f]) - etr[f], etr[f] - hi.predict(Xtr[f]))
            else:
                skip((~f).sum())
        lo, hi = fit(Xtr, etr)
        pl, ph = lo.predict(X[te]), hi.predict(X[te])
        cross += (pl > ph).sum(); n += te.sum()
        if full:
            q = calibration_quantile(oof, subj[tr], "subject")
            stored = pd.read_csv(f"results/intervals_{name}_cqr.csv")
            stored = stored[stored.subject == t]
            mine = np.maximum(df.hr_hat[te].to_numpy() + pl - q, HR_RANGE[0])
            print(f"   replay check {name} {t}: max |lo - stored| = {np.abs(mine - stored.lo_subject.to_numpy()).max():.2e}")
    return cross, n


dev = pd.read_csv("results/windows.csv")
rows = []
for name, d in [("DaLiA", dev[dev.dataset == "DaLiA"]), ("WESAD", dev[dev.dataset == "WESAD"]), ("DaLiA+WESAD", dev)]:
    c, n = replay_loso(d.reset_index(drop=True), name, check_first=(name == "DaLiA"))
    s = pd.read_csv(f"results/intervals_{name}_cqr.csv")
    empty = {w: int((s[f"lo_{w}"] > s[f"hi_{w}"]).sum()) for w in ("window", "subject")}
    rows.append({"setting": name, "crossing windows": c, "windows": n, "crossing %": 100 * c / n, **{f"empty ({w})": v for w, v in empty.items()}})
    print(name, "done", flush=True)

# GalaxyPPG: fresh generator, as step03_galaxyppg_external.py ran in its own process
rng = np.random.default_rng(0)
gal = pd.read_csv("results/windows_galaxy.csv")
X, e = dev[FEATURES].to_numpy(), (dev.hr_ref - dev.hr_hat).to_numpy()
subj, ds = dev.subject.to_numpy(), dev.dataset.to_numpy()
folds = person_folds(subj, ds)
oof = np.empty(len(dev))
for k in range(N_FOLDS):
    f = folds == k
    lo, hi = fit(X[~f], e[~f])
    oof[f] = np.maximum(lo.predict(X[f]) - e[f], e[f] - hi.predict(X[f]))
lo, hi = fit(X, e)
Xg = gal[FEATURES].to_numpy()
pl, ph = lo.predict(Xg), hi.predict(Xg)
empty = {}
for w in ("window", "subject"):
    q = calibration_quantile(oof, subj, w)
    a = np.maximum(gal.hr_hat + pl - q, HR_RANGE[0]); b = np.minimum(gal.hr_hat + ph + q, HR_RANGE[1])
    empty[f"empty ({w})"] = int((a > b).sum())
    cov = (gal.hr_ref >= a) & (gal.hr_ref <= b)
    per = cov.groupby(gal.subject).mean()
    print(f"   Galaxy replay {w}: at target {(per >= 0.9).sum()}/23, mean cov {per.mean():.4f}, median width {(b - a).median():.2f}")
rows.append({"setting": "GalaxyPPG", "crossing windows": int((pl > ph).sum()), "windows": len(gal),
             "crossing %": 100 * (pl > ph).mean(), **empty})
out = pd.DataFrame(rows)
out.to_csv("results/cqr_crossing.csv", index=False)
print(out.round(3).to_string(index=False))
