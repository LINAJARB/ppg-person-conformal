"""Step 15: calibration with motion-rich data, test on GalaxyPPG.

The scale model and the thresholds are fitted on PPG-DaLiA + WESAD + PPG-Sport
(left wrist; right wrist as a sensitivity analysis), then applied to GalaxyPPG.
MAST is unchanged. The motion-class edges stay those of PPG-DaLiA + WESAD.
Needs results/windows.csv, results/windows_galaxy.csv and
results/windows_ppgsport_{L,R}.csv (step 14).
Outputs: results/motionrich_summary.csv, motionrich_activities.csv,
motionrich_bootstrap.csv, motionrich_model.csv
"""
import numpy as np
import pandas as pd
from ppgs.conformal import METHODS, N_FOLDS, HR_RANGE, ALPHA, weighted_quantile, person_folds

F3 = ["snr_db", "motion", "disagree"]
DELTA, B = 0.10, 2000
rng = np.random.default_rng(2026)
GALMAP = {"baseline": "rest", "rest": "rest", "meditation": "rest", "adaptation": "rest", "screen reading": "rest",
          "standing": "rest", "TSST preparation": "stress", "TSST speech": "stress", "SSST preparation": "stress",
          "SSST singing": "stress", "keyboard typing": "typing", "mobile typing": "typing",
          "walking": "walking", "jogging": "jogging", "running": "running"}


def ceil_int(x):
    return int(np.ceil(round(x, 9)))


def q_window(s, subj):
    n = len(s); return weighted_quantile(s, np.ones(n), min(1.0, (1 - ALPHA) * (1 + 1 / n)))


def q_person(s, subj):
    _, inv, cnt = np.unique(subj, return_inverse=True, return_counts=True)
    lev = (1 - ALPHA) * (1 + 1 / len(cnt))
    return np.inf if lev > 1 else weighted_quantile(s, 1.0 / cnt[inv], lev)


def q_quantile(s, subj):
    r = np.sort([np.sort(s[subj == p])[ceil_int((1 - ALPHA) * (subj == p).sum()) - 1] for p in pd.unique(subj)])
    k = ceil_int((1 - DELTA) * (len(r) + 1))
    return np.inf if k > len(r) else r[k - 1]


dev = pd.read_csv("results/windows.csv")
gal = pd.read_csv("results/windows_galaxy.csv")
edges = np.quantile(dev.motion, [1 / 3, 2 / 3])
act = gal.condition.map(GALMAP)


def sport(w):
    s = pd.read_csv(f"results/windows_ppgsport_{w}.csv")
    s["subject"] = "X" + s.subject.str[1:]          # distinct from the S.. of PPG-DaLiA
    return s


settings = {"PPG-DaLiA + WESAD": dev,
            "+ PPG-Sport (left)": pd.concat([dev, sport("L")], ignore_index=True),
            "+ PPG-Sport (right)": pd.concat([dev, sport("R")], ignore_index=True)}

rows, act_rows, model_rows, cov_store, sig_store = [], [], [], {}, {}
for name, cal in settings.items():
    X, e = cal[F3].to_numpy(), (cal.hr_ref - cal.hr_hat).to_numpy()
    subj = cal.subject.to_numpy()
    f = person_folds(subj, cal.dataset.to_numpy())
    oof = np.empty(len(e))
    for k in range(N_FOLDS):
        oof[f == k] = METHODS["normalized"]().fit(X[f != k], e[f != k]).score(X[f == k], e[f == k])
    model = METHODS["normalized"]().fit(X, e)
    std = model.lr.coef_ * X.std(0)
    model_rows.append({"setting": name, "persons": len(np.unique(subj)), "windows": len(cal),
                       "coef x SD snr": std[0], "coef x SD motion": std[1], "coef x SD disagree": std[2],
                       "raw coef motion": model.lr.coef_[1],
                       "share of windows in motion class 3": np.mean(np.digitize(cal.motion, edges) == 2)})
    sig = model.sigma(gal[F3].to_numpy()); sig_store[name] = sig
    thr = {"window-level": q_window(oof, subj), "person-level": q_person(oof, subj),
           "person-quantile": q_quantile(oof, subj)}
    cls_cal, cls_te = np.digitize(cal.motion, edges), np.digitize(gal.motion, edges)
    qs = np.full(len(gal), np.nan)
    for c in np.unique(cls_te):
        qs[cls_te == c] = q_person(oof[cls_cal == c], subj[cls_cal == c])
    for rule, q in list(thr.items()) + [("motion-stratified", qs)]:
        lo = np.maximum(gal.hr_hat - q * sig, HR_RANGE[0]); hi = np.minimum(gal.hr_hat + q * sig, HR_RANGE[1])
        cov = ((gal.hr_ref >= lo) & (gal.hr_ref <= hi)).to_numpy(); w = (hi - lo).to_numpy()
        cov_store[(name, rule)] = cov
        per = pd.Series(cov).groupby(gal.subject.to_numpy()).mean()
        rows.append({"setting": name, "calibration": rule, "threshold": np.nanmean(q) if rule == "motion-stratified" else q,
                     "at target": int((per >= 1 - ALPHA).sum()), "persons": len(per), "worst": per.min(),
                     "mean cov": per.mean(), "median width": np.median(w), "mean width": w.mean()})
        for a in ["rest", "stress", "typing", "walking", "jogging", "running"]:
            sel = (act == a).to_numpy()
            act_rows.append({"setting": name, "calibration": rule, "activity": a, "coverage": cov[sel].mean(),
                             "median width": np.median(w[sel])})

summary = pd.DataFrame(rows); acts = pd.DataFrame(act_rows); models = pd.DataFrame(model_rows)

# Same median width as the motion-rich window-level intervals, but with the original scale model
ref = summary.set_index(["setting", "calibration"])
target_w = ref.loc[("+ PPG-Sport (left)", "window-level"), "median width"]
sig0 = sig_store["PPG-DaLiA + WESAD"]
grid = np.arange(1.0, 10.0, 0.001)
widths = [np.median(np.minimum(gal.hr_hat + q * sig0, 200) - np.maximum(gal.hr_hat - q * sig0, 40)) for q in grid]
qm = grid[int(np.argmin(np.abs(np.array(widths) - target_w)))]
lo = np.maximum(gal.hr_hat - qm * sig0, 40); hi = np.minimum(gal.hr_hat + qm * sig0, 200)
cov_m = ((gal.hr_ref >= lo) & (gal.hr_ref <= hi)).to_numpy()
cov_store[("PPG-DaLiA + WESAD", "same width")] = cov_m
per = pd.Series(cov_m).groupby(gal.subject.to_numpy()).mean()
summary = pd.concat([summary, pd.DataFrame([{"setting": "PPG-DaLiA + WESAD", "calibration": "same width",
                     "threshold": qm, "at target": int((per >= 0.9).sum()), "persons": len(per), "worst": per.min(),
                     "mean cov": per.mean(), "median width": np.median(hi - lo), "mean width": np.mean(hi - lo)}])])
for a in ["rest", "stress", "typing", "walking", "jogging", "running"]:
    sel = (act == a).to_numpy()
    acts = pd.concat([acts, pd.DataFrame([{"setting": "PPG-DaLiA + WESAD", "calibration": "same width", "activity": a,
                                           "coverage": cov_m[sel].mean(), "median width": np.median((hi - lo)[sel])}])])

# Participant bootstrap of GalaxyPPG (thresholds fixed): gain of the motion-rich calibration
ids = np.unique(gal.subject); n = len(ids)
idx = {s: np.where(gal.subject.to_numpy() == s)[0] for s in ids}
draws = rng.integers(0, n, size=(B, n))
pairs = [(("+ PPG-Sport (left)", r), ("PPG-DaLiA + WESAD", r)) for r in ["window-level", "person-level", "person-quantile"]]
pairs += [(("+ PPG-Sport (left)", "window-level"), ("PPG-DaLiA + WESAD", "person-quantile")),
          (("+ PPG-Sport (left)", "window-level"), ("PPG-DaLiA + WESAD", "same width"))]
brows = []
for a_key, b_key in pairs:
    for a in ["walking", "jogging", "running", "all"]:
        sel = np.ones(len(gal), bool) if a == "all" else (act == a).to_numpy()
        num_a = np.array([np.sum(cov_store[a_key][idx[s]] & sel[idx[s]]) for s in ids])
        num_b = np.array([np.sum(cov_store[b_key][idx[s]] & sel[idx[s]]) for s in ids])
        den = np.array([np.sum(sel[idx[s]]) for s in ids])
        est = (num_a.sum() - num_b.sum()) / den.sum()
        d = (num_a[draws].sum(1) - num_b[draws].sum(1)) / np.maximum(den[draws].sum(1), 1)
        lo_, hi_ = np.percentile(d, [2.5, 97.5])
        brows.append({"a": " / ".join(a_key), "b": " / ".join(b_key), "activity": a,
                      "difference (points)": 100 * est, "ci low": 100 * lo_, "ci high": 100 * hi_})
boot = pd.DataFrame(brows)

summary.to_csv("results/motionrich_summary.csv", index=False)
acts.to_csv("results/motionrich_activities.csv", index=False)
boot.to_csv("results/motionrich_bootstrap.csv", index=False)
models.to_csv("results/motionrich_model.csv", index=False)
pd.set_option("display.width", 250)
print(models.round(3).to_string(index=False))
print(summary.round(3).to_string(index=False))
print(acts.pivot_table(index=["setting", "calibration"], columns="activity", values="coverage").round(3))
print(acts.pivot_table(index=["setting", "calibration"], columns="activity", values="median width").round(1))
print(boot.round(1).to_string(index=False))
