"""Calibration rules, coverage-width curve, bootstrap, motion-stratified calibration and rejection (normalized score).

Per window, the leave-one-subject-out scale (sigma) and the thresholds of every
calibration are recomputed exactly as in step05_person_quantile.py, then:
  A. calibration rules: window-level, person-level, person-quantile, and the
     three methods of Dunn et al. (CDF pooling, single subsampling, repeated
     subsampling with averaged p-values, B = 100)
  B. coverage-width trade-off curve (one global threshold multiplying sigma)
  C. participant (cluster) bootstrap of the number at target and of paired
     differences, thresholds held fixed
  D. calibration stratified by motion level (Mondrian, person-level within class)
  E. MAE of retained windows when rejecting the widest intervals
  F. sensitivity on GalaxyPPG: without the 7 clock-shifted participants,
     without the 4 references below 40 bpm
Outputs in results/rules_*.csv and results/galaxy_sensitivity.csv
"""
import numpy as np
import pandas as pd
from ppgs.conformal import METHODS, N_FOLDS, HR_RANGE, ALPHA, weighted_quantile, person_folds

rng = np.random.default_rng(2026)
F3 = ["snr_db", "motion", "disagree"]
DELTA = 0.10
B_REP = 100
R_SINGLE = 50


def ceil_int(x):
    return int(np.ceil(round(x, 9)))


# ---------------- thresholds -----------------
def q_window(s, subj):
    n = len(s); lev = (1 - ALPHA) * (1 + 1 / n)
    return np.inf if lev > 1 else weighted_quantile(s, np.ones(n), lev)


def q_person(s, subj):
    _, inv, cnt = np.unique(subj, return_inverse=True, return_counts=True)
    lev = (1 - ALPHA) * (1 + 1 / len(cnt))
    return np.inf if lev > 1 else weighted_quantile(s, 1.0 / cnt[inv], lev)


def q_cdfpool(s, subj):           # Dunn et al., no finite-sample correction
    _, inv, cnt = np.unique(subj, return_inverse=True, return_counts=True)
    return weighted_quantile(s, 1.0 / cnt[inv], 1 - ALPHA)


def per_person_lists(s, subj):
    return [s[subj == p] for p in pd.unique(subj)]


def q_quantile(s, subj):
    r = np.sort([np.sort(v)[ceil_int((1 - ALPHA) * len(v)) - 1] for v in per_person_lists(s, subj)])
    k = ceil_int((1 - DELTA) * (len(r) + 1))
    return np.inf if k > len(r) else r[k - 1]


def q_single(s, subj, r):         # one window per person, split conformal on m values
    v = np.sort([g[r.integers(len(g))] for g in per_person_lists(s, subj)])
    k = ceil_int((1 - ALPHA) * (len(v) + 1))
    return np.inf if k > len(v) else v[k - 1]


def q_repeated(s, subj, r):       # average p-values over B subsamples, fixed score
    groups = per_person_lists(s, subj); m = len(groups)
    pool = np.concatenate([[g[r.integers(len(g))] for g in groups] for _ in range(B_REP)])
    need = B_REP * (ALPHA * (m + 1) - 1)          # count of pooled scores >= q required
    if need <= 0:
        return np.inf
    srt = np.sort(pool)[::-1]
    return srt[int(np.ceil(need)) - 1]           # largest q with N(>= q) >= need


RULES = {"window-level": q_window, "person-level": q_person, "person-quantile": q_quantile,
         "CDF pooling": q_cdfpool}


# ---------------- per-window table -----------------
def oof_model(X, e, subj, ds):
    f = person_folds(subj, ds)
    oof = np.empty(len(e))
    for k in range(N_FOLDS):
        oof[f == k] = METHODS["normalized"]().fit(X[f != k], e[f != k]).score(X[f == k], e[f == k])
    return oof, METHODS["normalized"]().fit(X, e)


def table_for(test, sig, thr, cal_scores=None, cal_subj=None, cal_motion=None, test_motion=None, mot_edges=None):
    out = test[["subject", "condition", "hr_hat", "hr_ref", "motion"]].copy()
    out["sigma"] = sig
    out["score"] = (test.hr_ref - test.hr_hat).abs().to_numpy() / sig
    for k, v in thr.items():
        out["q_" + k] = v
    if cal_scores is not None:      # Mondrian by motion class, person-level within class
        cls_cal = np.digitize(cal_motion, mot_edges); cls_te = np.digitize(test_motion, mot_edges)
        qc = np.full(len(test), np.nan)
        for c in np.unique(cls_te):
            sel = cls_cal == c
            qc[cls_te == c] = q_person(cal_scores[sel], cal_subj[sel])
        out["q_motion-stratified"] = qc
    return out


def thresholds(oof, subj):
    thr = {k: f(oof, subj) for k, f in RULES.items()}
    thr["single subsampling"] = np.mean([q_single(oof, subj, rng) for _ in range(R_SINGLE)])   # see note below
    thr["repeated subsampling"] = q_repeated(oof, subj, rng)
    return thr


def internal(df, edges):
    X, e = df[F3].to_numpy(), (df.hr_ref - df.hr_hat).to_numpy()
    subj, ds = df.subject.to_numpy(), df.dataset.to_numpy()
    parts, singles = [], []
    for t in np.unique(subj):
        tr, te = subj != t, subj == t
        oof, model = oof_model(X[tr], e[tr], subj[tr], ds[tr])
        sig = model.sigma(X[te])
        thr = thresholds(oof, subj[tr])
        parts.append(table_for(df[te], sig, thr, oof, subj[tr], df.motion.to_numpy()[tr], df.motion.to_numpy()[te], edges))
        singles.append([q_single(oof, subj[tr], rng) for _ in range(R_SINGLE)])
    return pd.concat(parts), singles


def external(dev, test, edges):
    X, e = dev[F3].to_numpy(), (dev.hr_ref - dev.hr_hat).to_numpy()
    subj = dev.subject.to_numpy()
    oof, model = oof_model(X, e, subj, dev.dataset.to_numpy())
    sig = model.sigma(test[F3].to_numpy())
    thr = thresholds(oof, subj)
    singles = [[q_single(oof, subj, rng) for _ in range(R_SINGLE)]]
    return table_for(test, sig, thr, oof, subj, dev.motion.to_numpy(), test.motion.to_numpy(), edges), singles


def covered(t, q):
    lo = np.maximum(t.hr_hat - q * t.sigma, HR_RANGE[0]); hi = np.minimum(t.hr_hat + q * t.sigma, HR_RANGE[1])
    return (t.hr_ref >= lo) & (t.hr_ref <= hi), hi - lo


def metrics(t, q):
    c, w = covered(t, q); per = c.groupby(t.subject).mean()
    return {"at target": int((per >= 0.9).sum()), "persons": len(per), "worst": per.min(), "mean cov": per.mean(),
            "median width": w.median(), "mean width": w.mean()}


dev = pd.read_csv("results/windows.csv"); gal = pd.read_csv("results/windows_galaxy.csv")
edges = np.quantile(dev.motion, [1 / 3, 2 / 3])           # motion terciles of the development windows
print("motion tercile edges (log10 g):", edges.round(3))

tables, singles = {}, {}
for name, run in [("PPG-DaLiA", lambda: internal(dev[dev.dataset == "DaLiA"].reset_index(drop=True), edges)),
                  ("WESAD", lambda: internal(dev[dev.dataset == "WESAD"].reset_index(drop=True), edges)),
                  ("Both", lambda: internal(dev, edges)),
                  ("GalaxyPPG", lambda: external(dev, gal, edges))]:
    tables[name], singles[name] = run()
    print(name, "done", flush=True)
pd.concat([t.assign(setting=k) for k, t in tables.items()]).to_csv("results/rules_windows.csv", index=False)

# A. calibration rules
rows = []
for name, t in tables.items():
    for rule in list(RULES) + ["repeated subsampling", "motion-stratified"]:
        rows.append({"setting": name, "rule": rule, **metrics(t, t["q_" + rule])})
    # single subsampling: average of metrics over R_SINGLE independent draws
    subj_order = list(pd.unique(t.subject)) if name != "GalaxyPPG" else None
    ms = []
    for r in range(R_SINGLE):
        if name == "GalaxyPPG":
            q = singles[name][0][r]
        else:
            qmap = {s: singles[name][i][r] for i, s in enumerate(sorted(t.subject.unique()))}
            q = t.subject.map(qmap)
        ms.append(metrics(t, q))
    m = pd.DataFrame(ms)
    rows.append({"setting": name, "rule": "single subsampling", **m.mean().to_dict(),
                 "at target min": m["at target"].min(), "at target max": m["at target"].max(),
                 "infinite threshold share": float(np.mean(np.isinf(np.concatenate(singles[name]))))})
rules = pd.DataFrame(rows); rules.to_csv("results/rules_summary.csv", index=False)
pd.set_option("display.width", 250)
print(rules.round(3).to_string(index=False))

# B. coverage-width curve
rows = []
for name, t in tables.items():
    for q in np.linspace(0.5, 12, 116):
        rows.append({"setting": name, "q": q, **metrics(t, q)})
curve = pd.DataFrame(rows); curve.to_csv("results/rules_curve.csv", index=False)

# C. participant bootstrap, thresholds held fixed
def per_person_cov(t, q):
    c, _ = covered(t, q); return c.groupby(t.subject).mean()

rows = []
B = 2000
for name in ["PPG-DaLiA+WESAD (separate)", "Both", "GalaxyPPG"]:
    if name.startswith("PPG-DaLiA+WESAD"):
        pp = {k: pd.concat([per_person_cov(tables[s], tables[s]["q_" + k]) for s in ["PPG-DaLiA", "WESAD"]])
              for k in ["window-level", "person-level", "person-quantile"]}
    else:
        t = tables[name]; pp = {k: per_person_cov(t, t["q_" + k]) for k in ["window-level", "person-level", "person-quantile"]}
    ids = pp["window-level"].index.to_numpy(); n = len(ids)
    at = {k: (v.reindex(ids) >= 0.9).to_numpy().astype(int) for k, v in pp.items()}
    boot = rng.integers(0, n, size=(B, n))
    res = {k: at[k][boot].sum(1) for k in at}
    for k in at:
        lo, hi = np.percentile(res[k], [2.5, 97.5])
        rows.append({"setting": name, "quantity": f"at target, {k}", "estimate": at[k].sum(), "n": n, "ci low": lo, "ci high": hi})
    for a, b in [("person-level", "window-level"), ("person-quantile", "person-level"), ("person-quantile", "window-level")]:
        d = res[a] - res[b]; lo, hi = np.percentile(d, [2.5, 97.5])
        rows.append({"setting": name, "quantity": f"difference {a} - {b}", "estimate": at[a].sum() - at[b].sum(), "n": n,
                     "ci low": lo, "ci high": hi, "share > 0": float(np.mean(d > 0))})
boot = pd.DataFrame(rows); boot.to_csv("results/rules_bootstrap.csv", index=False)
print(boot.round(3).to_string(index=False))

# D. per-activity coverage for the motion-stratified rule vs person-level
galmap = {"baseline": "rest", "rest": "rest", "meditation": "rest", "adaptation": "rest", "screen reading": "rest",
          "standing": "rest", "TSST preparation": "stress", "TSST speech": "stress", "SSST preparation": "stress",
          "SSST singing": "stress", "keyboard typing": "typing", "mobile typing": "typing",
          "walking": "walking", "jogging": "jogging", "running": "running"}
rows = []
for name, t in tables.items():
    cond = t.condition.map(galmap) if name == "GalaxyPPG" else t.condition
    for rule in ["person-level", "person-quantile", "motion-stratified"]:
        c, w = covered(t, t["q_" + rule])
        for k, g in c.groupby(cond):
            rows.append({"setting": name, "rule": rule, "condition": k, "coverage": g.mean(), "median width": w[g.index].median()})
act = pd.DataFrame(rows); act.to_csv("results/rules_activity.csv", index=False)
print(act[act.setting == "GalaxyPPG"].pivot(index="condition", columns="rule", values=["coverage", "median width"]).round(3))

# E. error of retained windows when rejecting the widest intervals (sigma ranking)
rows = []
for name, t in tables.items():
    err = (t.hr_hat - t.hr_ref).abs().to_numpy(); order = np.argsort(t.sigma.to_numpy())
    for frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        keep = order[: int(round(frac * len(t)))]
        rows.append({"setting": name, "retained": frac, "MAE retained": err[keep].mean(), "MAE all": err.mean()})
bel = gal.merge(pd.read_csv("results/beliefppg_galaxy.csv"), on=["subject", "window"])
eb = (bel.hr_belief - bel.hr_ref).abs().to_numpy(); ob = np.argsort(bel.unc_belief.to_numpy())
for frac in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
    keep = ob[: int(round(frac * len(bel)))]
    rows.append({"setting": "BeliefPPG (GalaxyPPG)", "retained": frac, "MAE retained": eb[keep].mean(), "MAE all": eb.mean()})
util = pd.DataFrame(rows); util.to_csv("results/rules_rejection.csv", index=False)
print(util.round(2).to_string(index=False))

# F. GalaxyPPG sensitivity
t = tables["GalaxyPPG"]
gp = pd.read_csv("results/galaxy_participants.csv")
shifted = {"G" + str(int(x[1:])) for x in gp.loc[gp["offset applied"], "participant"]}
print("shifted participants:", sorted(shifted))
rows = []
for lab, sel in [("all", np.ones(len(t), bool)), ("without clock-shifted participants", ~t.subject.astype(str).isin(shifted)),
                 ("without references below 40 bpm", t.hr_ref >= 40)]:
    for rule in ["window-level", "person-level", "person-quantile"]:
        rows.append({"subset": lab, "rule": rule, **metrics(t[sel], t["q_" + rule][sel])})
sens = pd.DataFrame(rows); sens.to_csv("results/galaxy_sensitivity.csv", index=False)
print(sens.round(3).to_string(index=False))
