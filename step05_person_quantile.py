"""Person-quantile calibration (Proposition 2 of the paper).

Target: with probability at least 1 - delta, a new person has at least 1 - alpha
of its windows covered. For each calibration person i, r_i is the
ceil((1 - alpha) n_i)-th smallest of its scores. The threshold is the k-th
smallest r_i, k = ceil((1 - delta)(m + 1)), and +inf if k > m.

Compared with person-level calibration (Proposition 1, average of the
per-person score distributions at level (1 - alpha)(1 + 1/m)).

Outputs (results/):
  person_quantile_main.csv      LOSO on PPG-DaLiA, WESAD, both, normalized score
  person_quantile_galaxy.csv    GalaxyPPG external, three scores (CQR replayed exactly)
  person_quantile_belief.csv    BeliefPPG within GalaxyPPG
  person_quantile_split.csv     split version, 20 partitions
  person_quantile_folds.csv     five random fold assignments
  person_quantile_phases.csv    coverage per condition (pooled windows)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from ppgs.conformal import (METHODS, N_FOLDS, HR_RANGE, ALPHA, CQR_MAX_FIT,
                            weighted_quantile, person_folds)

DELTA = 0.10
F3 = ["snr_db", "motion", "disagree"]


def ceil_int(x):
    return int(np.ceil(round(x, 9)))


def person_quantile(scores, subj, delta=DELTA):
    r = []
    for s in pd.unique(subj):
        v = np.sort(scores[subj == s])
        r.append(v[ceil_int((1 - ALPHA) * len(v)) - 1])
    r = np.sort(r)
    k = ceil_int((1 - delta) * (len(r) + 1))
    return np.inf if k > len(r) else r[k - 1]


def mixture_quantile(scores, subj):
    _, inv, counts = np.unique(subj, return_inverse=True, return_counts=True)
    level = (1 - ALPHA) * (1 + 1 / len(counts))
    return np.inf if level > 1 else weighted_quantile(scores, 1.0 / counts[inv], level)


QUANTILES = {"person-level": mixture_quantile, "person-quantile": person_quantile}


def folds_random(subj, ds, seed):
    rng = np.random.default_rng(seed)
    fold = {}
    for d in sorted(set(ds)):
        p = sorted(set(subj[ds == d]), key=lambda v: int(v[1:]))
        for s in rng.permutation(p):
            fold[s] = len(fold) % N_FOLDS   # balanced, random order
    return np.array([fold[s] for s in subj])


def oof_model(X, e, subj, ds, seed=None):
    f = person_folds(subj, ds) if seed is None else folds_random(subj, ds, seed)
    oof = np.empty(len(e))
    for k in range(N_FOLDS):
        oof[f == k] = METHODS["normalized"]().fit(X[f != k], e[f != k]).score(X[f == k], e[f == k])
    return oof, METHODS["normalized"]().fit(X, e)


def apply(test, lo_off, hi_off):
    lo = np.maximum(test.hr_hat.to_numpy() + lo_off, HR_RANGE[0])
    hi = np.minimum(test.hr_hat.to_numpy() + hi_off, HR_RANGE[1])
    y = test.hr_ref.to_numpy()
    return pd.DataFrame({"subject": test.subject.to_numpy(), "condition": test.condition.to_numpy(),
                         "cov": (y >= lo) & (y <= hi), "width": hi - lo})


def metrics(res):
    per = res.groupby("subject")["cov"].mean()
    return {"at target": int((per >= 1 - ALPHA).sum()), "persons": len(per), "mean cov": per.mean(),
            "pooled cov": res["cov"].mean(), "worst": per.min(),
            "median width": res.width.median(), "mean width": res.width.mean()}


def internal(df, feats, seed=None):
    X, e = df[feats].to_numpy(), (df.hr_ref - df.hr_hat).to_numpy()
    subj, ds = df.subject.to_numpy(), df.dataset.to_numpy()
    out = {k: [] for k in QUANTILES}
    for t in np.unique(subj):
        tr, te = subj != t, subj == t
        oof, model = oof_model(X[tr], e[tr], subj[tr], ds[tr], seed)
        s = model.sigma(X[te])
        for k, qf in QUANTILES.items():
            q = qf(oof, subj[tr])
            out[k].append(apply(df[te], -q * s, q * s))
    return {k: pd.concat(v) for k, v in out.items()}


dev = pd.read_csv("results/windows.csv")
gal = pd.read_csv("results/windows_galaxy.csv")
galmap = {"baseline": "rest", "rest": "rest", "meditation": "rest", "adaptation": "rest", "screen reading": "rest",
          "standing": "rest", "TSST preparation": "stress", "TSST speech": "stress", "SSST preparation": "stress",
          "SSST singing": "stress", "keyboard typing": "typing", "mobile typing": "typing",
          "walking": "walking", "jogging": "jogging", "running": "running"}
pd.set_option("display.width", 250)

# 1. leave-one-subject-out on the development data, normalized score
rows, phases = [], []
for name, d in [("PPG-DaLiA", dev[dev.dataset == "DaLiA"]), ("WESAD", dev[dev.dataset == "WESAD"]), ("Both", dev)]:
    res = internal(d.reset_index(drop=True), F3)
    for k, r in res.items():
        rows.append({"setting": name, "calibration": k, **metrics(r)})
        if name != "Both":
            for c, g in r.groupby("condition"):
                phases.append({"setting": name, "calibration": k, "condition": c, "coverage": g["cov"].mean(),
                               "median width": g.width.median()})
        if name == "WESAD":
            s11 = r[(r.subject == "W11")]
            print(k, "S11 overall", round(s11["cov"].mean(), 4), "stress", round(s11[s11.condition == "stress"]["cov"].mean(), 4))
main = pd.DataFrame(rows); main.to_csv("results/person_quantile_main.csv", index=False)
print(main.round(4).to_string(index=False), flush=True)

import sys
if "--folds-only" in sys.argv:
    galaxy = None
# 2. GalaxyPPG external, three scores. CQR replays step03_galaxyppg_external.py (fresh generator, 5 fold fits then final fit)
X, e = dev[F3].to_numpy(), (dev.hr_ref - dev.hr_hat).to_numpy()
subj, ds = dev.subject.to_numpy(), dev.dataset.to_numpy()
f = person_folds(subj, ds)
Xg = gal[F3].to_numpy()
rows = []
for method in ["constant", "normalized", "cqr"]:
    if method == "cqr":
        rng = np.random.default_rng(0)
        def fit(Xa, ea):
            idx = rng.choice(len(Xa), min(len(Xa), CQR_MAX_FIT), replace=False)
            return (QuantileRegressor(quantile=ALPHA / 2, alpha=0.0, solver="highs").fit(Xa[idx], ea[idx]),
                    QuantileRegressor(quantile=1 - ALPHA / 2, alpha=0.0, solver="highs").fit(Xa[idx], ea[idx]))
        oof = np.empty(len(dev))
        for k in range(N_FOLDS):
            lo_m, hi_m = fit(X[f != k], e[f != k])
            oof[f == k] = np.maximum(lo_m.predict(X[f == k]) - e[f == k], e[f == k] - hi_m.predict(X[f == k]))
        lo_m, hi_m = fit(X, e)
        base_lo, base_hi, scale = lo_m.predict(Xg), hi_m.predict(Xg), 1.0
    else:
        oof = np.empty(len(dev))
        for k in range(N_FOLDS):
            oof[f == k] = METHODS[method]().fit(X[f != k], e[f != k]).score(X[f == k], e[f == k])
        model = METHODS[method]().fit(X, e)
        base_lo = base_hi = np.zeros(len(gal))
        scale = model.sigma(Xg) if method == "normalized" else 1.0
    for k, qf in QUANTILES.items():
        q = qf(oof, subj)
        r = apply(gal, base_lo - q * scale, base_hi + q * scale)
        rows.append({"score": method, "calibration": k, **metrics(r)})
        if method == "normalized":
            grp = r.condition.map(galmap)
            for c, g in r.groupby(grp):
                phases.append({"setting": "GalaxyPPG", "calibration": k, "condition": c, "coverage": g["cov"].mean(),
                               "median width": g.width.median()})
            keep = ~r.condition.isin(["jogging", "running"])
            m = metrics(r[keep])
            print(k, "Galaxy without jogging/running:", m["at target"], round(m["mean cov"], 4))
galaxy = pd.DataFrame(rows); galaxy.to_csv("results/person_quantile_galaxy.csv", index=False)
print(galaxy.round(4).to_string(index=False), flush=True)
pd.DataFrame(phases).to_csv("results/person_quantile_phases.csv", index=False)

# 3. BeliefPPG within GalaxyPPG
bel = gal.merge(pd.read_csv("results/beliefppg_galaxy.csv"), on=["subject", "window"])
bel["hr_hat"] = bel.hr_belief
bel["log_unc"] = np.log(bel.unc_belief)
res = internal(bel, ["log_unc"])
belief = pd.DataFrame([{"calibration": k, **metrics(r)} for k, r in res.items()])
for k, r in res.items():
    print("BeliefPPG", k, "running coverage", round(r[r.condition == "running"]["cov"].mean(), 4))
belief.to_csv("results/person_quantile_belief.csv", index=False)
print(belief.round(4).to_string(index=False), flush=True)

# 4. split version: 10 fitting persons, the others calibrate, 20 partitions (same design as review_split)
def split_run(train, test, rng):
    persons = np.array(sorted(train.subject.unique()))
    fit_p = rng.choice(persons, 10, replace=False)
    fit, cal = train[train.subject.isin(fit_p)], train[~train.subject.isin(fit_p)]
    model = METHODS["normalized"]().fit(fit[F3].to_numpy(), (fit.hr_ref - fit.hr_hat).to_numpy())
    s = model.score(cal[F3].to_numpy(), (cal.hr_ref - cal.hr_hat).to_numpy())
    sig = model.sigma(test[F3].to_numpy())
    return {k: apply(test, -qf(s, cal.subject.to_numpy()) * sig, qf(s, cal.subject.to_numpy()) * sig)
            for k, qf in QUANTILES.items()}

rng = np.random.default_rng(0)
rows = []
for rep in range(20):
    parts = {k: [] for k in QUANTILES}
    for t in dev.subject.unique():
        for k, r in split_run(dev[dev.subject != t], dev[dev.subject == t], rng).items():
            parts[k].append(r)
    for k in QUANTILES:
        rows.append({"setting": "Both (LOSO)", "rep": rep, "calibration": k, **metrics(pd.concat(parts[k]))})
    for k, r in split_run(dev, gal, rng).items():
        rows.append({"setting": "GalaxyPPG", "rep": rep, "calibration": k, **metrics(r)})
split = pd.DataFrame(rows); split.to_csv("results/person_quantile_split.csv", index=False)
print(split.groupby(["setting", "calibration"]).agg(at=("at target", "mean"), at_min=("at target", "min"),
      at_max=("at target", "max"), cov=("mean cov", "mean"), worst=("worst", "mean"),
      wmed=("median width", "mean")).round(3), flush=True)

# 5. fold sensitivity
rows = []
for name, d in [("PPG-DaLiA", dev[dev.dataset == "DaLiA"]), ("WESAD", dev[dev.dataset == "WESAD"])]:
    for seed in range(5):
        for k, r in internal(d.reset_index(drop=True), F3, seed=seed).items():
            rows.append({"setting": name, "seed": seed, "calibration": k, **metrics(r)})
folds = pd.DataFrame(rows); folds.to_csv("results/person_quantile_folds.csv", index=False)
print(folds.round(3).to_string(index=False))
