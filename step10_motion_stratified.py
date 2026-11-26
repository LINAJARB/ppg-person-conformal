"""Further analyses of the motion-stratified (Mondrian) variant.

  A. participant bootstrap of the number at target for the motion-stratified rule,
     and of its paired difference with person-level calibration (thresholds fixed)
  B. participant bootstrap of the coverage per activity (pooled over windows), for
     person-level and motion-stratified calibration, and of their difference
  C. median width per motion class and share of windows per class
  D. motion-stratified calibration applied to BeliefPPG within GalaxyPPG (LOSO)
Inputs: results/rules_windows.csv (from step09_calibration_rules.py), windows*.csv
Outputs: results/stratified_*.csv
"""
import numpy as np
import pandas as pd
from ppgs.conformal import METHODS, N_FOLDS, HR_RANGE, ALPHA, weighted_quantile, person_folds

rng = np.random.default_rng(7)
B = 2000
GALMAP = {"baseline": "rest", "rest": "rest", "meditation": "rest", "adaptation": "rest", "screen reading": "rest",
          "standing": "rest", "TSST preparation": "stress", "TSST speech": "stress", "SSST preparation": "stress",
          "SSST singing": "stress", "keyboard typing": "typing", "mobile typing": "typing",
          "walking": "walking", "jogging": "jogging", "running": "running"}
RULES = ["person-level", "motion-stratified"]


def covered(t, q):
    lo = np.maximum(t.hr_hat - q * t.sigma, HR_RANGE[0]); hi = np.minimum(t.hr_hat + q * t.sigma, HR_RANGE[1])
    return ((t.hr_ref >= lo) & (t.hr_ref <= hi)).to_numpy(), (hi - lo).to_numpy()


def q_person(s, subj):
    _, inv, cnt = np.unique(subj, return_inverse=True, return_counts=True)
    lev = (1 - ALPHA) * (1 + 1 / len(cnt))
    return np.inf if lev > 1 else weighted_quantile(s, 1.0 / cnt[inv], lev)


tw = pd.read_csv("results/rules_windows.csv")
dev = pd.read_csv("results/windows.csv")
edges = np.quantile(dev.motion, [1 / 3, 2 / 3])

# A and B
rowsA, rowsB, rowsC = [], [], []
for name in ["Both", "GalaxyPPG"]:
    t = tw[tw.setting == name].reset_index(drop=True)
    t["act"] = t.condition.map(GALMAP).fillna("transition") if name == "GalaxyPPG" else t.condition
    ids = np.array(sorted(t.subject.unique())); n = len(ids)
    cov = {r: covered(t, t["q_" + r]) for r in RULES}
    per = {r: pd.Series(cov[r][0]).groupby(t.subject).mean().reindex(ids).to_numpy() for r in RULES}
    boot = rng.integers(0, n, size=(B, n))
    at = {r: (per[r] >= 0.9).astype(int) for r in RULES}
    res = {r: at[r][boot].sum(1) for r in RULES}
    for r in RULES:
        lo, hi = np.percentile(res[r], [2.5, 97.5])
        rowsA.append({"setting": name, "quantity": f"at target, {r}", "estimate": at[r].sum(), "n": n, "ci low": lo, "ci high": hi})
    d = res["motion-stratified"] - res["person-level"]; lo, hi = np.percentile(d, [2.5, 97.5])
    rowsA.append({"setting": name, "quantity": "difference motion-stratified - person-level",
                  "estimate": at["motion-stratified"].sum() - at["person-level"].sum(), "n": n, "ci low": lo, "ci high": hi})
    # B. per activity, pooled coverage over windows of the resampled participants
    for a in sorted(t.act.unique()):
        sel = (t.act == a).to_numpy()
        nwin = pd.Series(sel.astype(int)).groupby(t.subject).sum().reindex(ids).to_numpy()
        hits = {r: pd.Series((cov[r][0] & sel).astype(int)).groupby(t.subject).sum().reindex(ids).to_numpy() for r in RULES}
        N = nwin[boot].sum(1)
        ok = N > 0
        bc = {r: hits[r][boot].sum(1)[ok] / N[ok] for r in RULES}
        row = {"setting": name, "activity": a, "windows": int(nwin.sum()), "persons": int((nwin > 0).sum())}
        for r in RULES:
            row[f"{r} cov"] = hits[r].sum() / nwin.sum()
            row[f"{r} lo"], row[f"{r} hi"] = np.percentile(bc[r], [2.5, 97.5])
            row[f"{r} median width"] = np.median(cov[r][1][sel])
        dd = bc["motion-stratified"] - bc["person-level"]
        row["diff"] = row["motion-stratified cov"] - row["person-level cov"]
        row["diff lo"], row["diff hi"] = np.percentile(dd, [2.5, 97.5])
        rowsB.append(row)
    # C. width per motion class
    cls = np.digitize(t.motion.to_numpy(), edges)
    for c in range(3):
        sel = cls == c
        rowsC.append({"setting": name, "motion class": c, "share of windows": sel.mean(),
                      **{f"{r} median width": np.median(cov[r][1][sel]) for r in RULES},
                      **{f"{r} coverage": cov[r][0][sel].mean() for r in RULES}})
    rowsC.append({"setting": name, "motion class": "all", "share of windows": 1.0,
                  **{f"{r} median width": np.median(cov[r][1]) for r in RULES},
                  **{f"{r} mean width": np.mean(cov[r][1]) for r in RULES},
                  **{f"{r} coverage": cov[r][0].mean() for r in RULES}})
pd.set_option("display.width", 250)
A = pd.DataFrame(rowsA); A.to_csv("results/stratified_bootstrap.csv", index=False); print(A.round(2).to_string(index=False))
Bt = pd.DataFrame(rowsB); Bt.to_csv("results/stratified_activity_bootstrap.csv", index=False); print(Bt.round(3).to_string(index=False))
C = pd.DataFrame(rowsC); C.to_csv("results/stratified_width_by_class.csv", index=False); print(C.round(3).to_string(index=False))

# D. BeliefPPG within GalaxyPPG, LOSO, scale model on log posterior SD
gal = pd.read_csv("results/windows_galaxy.csv")
bel = gal.merge(pd.read_csv("results/beliefppg_galaxy.csv"), on=["subject", "window"])
bel["hr_hat"] = bel.hr_belief
X = np.log(bel.unc_belief.to_numpy())[:, None]
e = (bel.hr_ref - bel.hr_hat).to_numpy()
subj = bel.subject.to_numpy(); ds = bel.dataset.to_numpy() if "dataset" in bel else np.array(["GalaxyPPG"] * len(bel))
mot = bel.motion.to_numpy(); cls = np.digitize(mot, edges)
parts = []
for p in np.unique(subj):
    tr, te = subj != p, subj == p
    f = person_folds(subj[tr], ds[tr])
    oof = np.empty(tr.sum())
    for k in range(N_FOLDS):
        oof[f == k] = METHODS["normalized"]().fit(X[tr][f != k], e[tr][f != k]).score(X[tr][f == k], e[tr][f == k])
    sig = METHODS["normalized"]().fit(X[tr], e[tr]).sigma(X[te])
    out = bel.loc[te, ["subject", "condition", "hr_hat", "hr_ref"]].copy(); out["sigma"] = sig
    out["q_person-level"] = q_person(oof, subj[tr])
    qc = np.full(te.sum(), np.nan)
    for c in np.unique(cls[te]):
        s = cls[tr] == c
        qc[cls[te] == c] = q_person(oof[s], subj[tr][s])
    out["q_motion-stratified"] = qc
    parts.append(out)
bt = pd.concat(parts).reset_index(drop=True)
bt["act"] = bt.condition.map(GALMAP).fillna("transition")
rows = []
for r in RULES:
    c, w = covered(bt, bt["q_" + r])
    per = pd.Series(c).groupby(bt.subject).mean()
    row = {"rule": r, "at target": int((per >= 0.9).sum()), "persons": len(per), "worst": per.min(),
           "mean cov": per.mean(), "median width": np.median(w), "mean width": np.mean(w)}
    for a in ["rest", "stress", "typing", "walking", "jogging", "running"]:
        sel = (bt.act == a).to_numpy()
        row[f"{a} cov"] = c[sel].mean(); row[f"{a} width"] = np.median(w[sel])
    rows.append(row)
D = pd.DataFrame(rows); D.to_csv("results/stratified_belief.csv", index=False); print(D.round(3).T.to_string())
