"""Ablation for the revised paper: weighting (window / person) x finite-sample
correction counted in windows (n) or in persons (m), normalized score.

Settings: PPG-DaLiA, WESAD, both pooled (leave-one-subject-out, nested
cross-fitting), GalaxyPPG external (calibration on the 30 development
persons), and BeliefPPG within GalaxyPPG (leave-one-subject-out, scale from
log of BeliefPPG's posterior SD).
"""
import numpy as np
import pandas as pd
from ppgs.conformal import METHODS, N_FOLDS, HR_RANGE, ALPHA, weighted_quantile

VARIANTS = [("window", "n"), ("person", "n"), ("window", "m"), ("person", "m")]


def quantile(scores, subjects, weighting, count):
    _, inv, counts = np.unique(subjects, return_inverse=True, return_counts=True)
    w = np.ones(len(scores)) if weighting == "window" else 1.0 / counts[inv]
    k = len(scores) if count == "n" else len(counts)
    level = (1 - ALPHA) * (1 + 1 / k)
    return np.inf if level > 1 else weighted_quantile(scores, w, level)


def folds(subj, ds):
    fold = {}
    for d in sorted(set(ds)):
        for s in sorted(set(subj[ds == d]), key=lambda v: int(v[1:])):
            fold[s] = len(fold) % N_FOLDS
    return np.array([fold[s] for s in subj])


def fit_and_scores(X, e, subj, ds):
    f = folds(subj, ds)
    oof = np.empty(len(e))
    for k in range(N_FOLDS):
        oof[f == k] = METHODS["normalized"]().fit(X[f != k], e[f != k]).score(X[f == k], e[f == k])
    return oof, METHODS["normalized"]().fit(X, e)


def apply(test, X_test, model, q):
    s = model.sigma(X_test)
    lo = np.maximum(test.hr_hat.to_numpy() - q * s, HR_RANGE[0])
    hi = np.minimum(test.hr_hat.to_numpy() + q * s, HR_RANGE[1])
    y = test.hr_ref.to_numpy()
    return pd.DataFrame({"subject": test.subject.to_numpy(), "cov": (y >= lo) & (y <= hi), "width": hi - lo})


def metrics(res):
    per = res.groupby("subject")["cov"].mean()
    return {"coverage (windows)": res["cov"].mean(), "coverage (mean over persons)": per.mean(),
            "at target": int((per >= 1 - ALPHA).sum()), "persons": len(per), "worst person": per.min(),
            "median width": res.width.median(), "mean width": res.width.mean()}


def internal(df, feats):
    X, e = df[feats].to_numpy(), (df.hr_ref - df.hr_hat).to_numpy()
    subj, ds = df.subject.to_numpy(), df.dataset.to_numpy()
    out = {v: [] for v in VARIANTS}
    for t in np.unique(subj):
        tr, te = subj != t, subj == t
        oof, model = fit_and_scores(X[tr], e[tr], subj[tr], ds[tr])
        for v in VARIANTS:
            out[v].append(apply(df[te], X[te], model, quantile(oof, subj[tr], *v)))
    return {v: metrics(pd.concat(out[v])) for v in VARIANTS}


def external(dev, test, feats):
    X, e = dev[feats].to_numpy(), (dev.hr_ref - dev.hr_hat).to_numpy()
    subj = dev.subject.to_numpy()
    oof, model = fit_and_scores(X, e, subj, dev.dataset.to_numpy())
    return {v: metrics(apply(test, test[feats].to_numpy(), model, quantile(oof, subj, *v))) for v in VARIANTS}


F3 = ["snr_db", "motion", "disagree"]
dev = pd.read_csv("results/windows.csv")
gal = pd.read_csv("results/windows_galaxy.csv")
b = pd.read_csv("results/beliefppg_galaxy.csv")
bel = gal.merge(b, on=["subject", "window"], how="inner")
bel["hr_hat"] = bel.hr_belief
bel["log_unc"] = np.log(bel.unc_belief)

rows = []
settings = [("PPG-DaLiA", lambda: internal(dev[dev.dataset == "DaLiA"].reset_index(drop=True), F3)),
            ("WESAD", lambda: internal(dev[dev.dataset == "WESAD"].reset_index(drop=True), F3)),
            ("Both", lambda: internal(dev, F3)),
            ("GalaxyPPG (external)", lambda: external(dev, gal, F3)),
            ("BeliefPPG on GalaxyPPG", lambda: internal(bel, ["log_unc"]))]
for name, run in settings:
    for v, m in run().items():
        rows.append({"setting": name, "weighting": v[0], "correction": v[1], **m})
    print(name, "done", flush=True)
abl = pd.DataFrame(rows)
abl.to_csv("results/ablation.csv", index=False)
pd.set_option("display.width", 250)
print(abl.round(4).to_string(index=False))
