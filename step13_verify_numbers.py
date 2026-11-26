"""Recomputes every number stated in the paper and the supplementary material
from the result files, and compares it with the value written in the text.
Each line prints OK or MISMATCH with the stated and recomputed values.
Values that come from earlier runs not stored here are listed at the end."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

R = "results/"
ok = bad = 0


def check(label, stated, value, ndigits=0):
    global ok, bad
    v = round(float(value), ndigits)
    good = abs(v - stated) < 1e-9
    ok += good; bad += not good
    print(f"{'OK      ' if good else 'MISMATCH'} {label}: stated {stated}, recomputed {value:.4f}")


def cov(r, w):
    return (r.hr_ref >= r[f"lo_{w}"]) & (r.hr_ref <= r[f"hi_{w}"])


def stats(r, w):
    c = cov(r, w); p = c.groupby(r.subject).mean(); wd = r[f"hi_{w}"] - r[f"lo_{w}"]
    return dict(at=int((p >= 0.9).sum()), n=len(p), worst=100 * p.min(), mean=100 * p.mean(),
                pooled=100 * c.mean(), med=wd.median(), meanw=wd.mean(), per=p)


dev = pd.read_csv(R + "windows.csv"); gal = pd.read_csv(R + "windows_galaxy.csv")
iv = {(s, m): pd.read_csv(R + f"intervals_{f}_{m}.csv") for s, f in
      [("PPG-DaLiA", "DaLiA"), ("WESAD", "WESAD"), ("Both", "DaLiA+WESAD")] for m in ["constant", "normalized", "cqr"]}
ig = pd.read_csv(R + "intervals_galaxy_normalized.csv"); ib = pd.read_csv(R + "intervals_galaxy_beliefppg.csv")
pqm = pd.read_csv(R + "person_quantile_main.csv"); pqg = pd.read_csv(R + "person_quantile_galaxy.csv")
pqb = pd.read_csv(R + "person_quantile_belief.csv"); pqs = pd.read_csv(R + "person_quantile_split.csv")
pqf = pd.read_csv(R + "person_quantile_folds.csv"); pqp = pd.read_csv(R + "person_quantile_phases.csv")
abl = pd.read_csv(R + "ablation.csv"); sg = pd.read_csv(R + "summary_galaxy.csv")
sb = pd.read_csv(R + "summary_beliefppg_galaxy.csv"); cq = pd.read_csv(R + "cqr_crossing.csv")

print("== Data")
check("development windows", 107811, len(dev)); check("PPG-DaLiA windows", 64697, (dev.dataset == "DaLiA").sum())
check("WESAD windows", 43114, (dev.dataset == "WESAD").sum()); check("GalaxyPPG windows", 43128, len(gal))
check("GalaxyPPG participants", 23, gal.subject.nunique())
check("dev reference min", 42, dev.hr_ref.min()); check("dev reference max", 187, dev.hr_ref.max())
check("GalaxyPPG windows < 40 bpm", 4, (gal.hr_ref < 40).sum())
check("their min", 32, gal.hr_ref[gal.hr_ref < 40].min()); check("their max", 36, gal.hr_ref[gal.hr_ref < 40].max())
print("   conditions of these windows:", gal.condition[gal.hr_ref < 40].unique())
gp = pd.read_csv(R + "galaxy_participants.csv")
check("clock shifted for participants", 7, gp["offset applied"].sum())
check("max applied shift (s)", 3.2, gp.loc[gp["offset applied"], "chest clock offset (s)"].abs().max(), 1)
check("disagreement = 60 (% dev windows)", 3.4, 100 * (dev.disagree >= 60).mean(), 1)
n_i = dev.groupby("subject").size(); check("max/min recording length (about two)", 2.0, n_i.max() / n_i.min(), 1)

print("== Estimator accuracy")
for name, d, sm, ss in [("PPG-DaLiA", dev[dev.dataset == "DaLiA"], 9.9, 4.0), ("WESAD", dev[dev.dataset == "WESAD"], 7.3, 2.4),
                        ("GalaxyPPG", gal, 8.3, 3.4)]:
    m = (d.hr_hat - d.hr_ref).abs().groupby(d.subject).mean()
    check(f"MAE {name} mean", sm, m.mean(), 1); check(f"MAE {name} SD", ss, m.std(), 1)
dl = dev[dev.dataset == "DaLiA"]; ae = (dl.hr_hat - dl.hr_ref).abs()
for c, v in [("stairs", 29), ("table soccer", 23), ("cycling", 18), ("walking", 18)]:
    check(f"MAE {c}", v, ae[dl.condition == c].mean())
ws = dev[dev.dataset == "WESAD"]; aw = (ws.hr_hat - ws.hr_ref).abs()
check("MAE WESAD stress", 21, aw[ws.condition == "stress"].mean())
rest = pd.concat([ae[dl.condition == "sitting"], aw[ws.condition == "baseline"]])
print(f"   MAE at rest: DaLiA sitting {ae[dl.condition == 'sitting'].mean():.2f}, WESAD baseline {aw[ws.condition == 'baseline'].mean():.2f}, "
      f"meditation {aw[ws.condition == 'meditation'].mean():.2f}, amusement {aw[ws.condition == 'amusement'].mean():.2f} (stated 2 to 4)")
from sklearn.linear_model import LinearRegression
X = dev[["snr_db", "motion", "disagree"]].to_numpy(); y = np.log((dev.hr_ref - dev.hr_hat).abs() + 1)
lr = LinearRegression().fit(X, y); check("scale model R2 (%)", 38, 100 * lr.score(X, y))
std = lr.coef_ * X.std(0)   # change of log(|e|+1) per SD of each descriptor
check("std coef disagreement", 0.48, std[2], 2); check("std coef SNR", -0.20, std[0], 2); check("std coef motion", 0.12, std[1], 2)

print("== Development results, normalized score")
T2 = {"PPG-DaLiA": [(9, 80.4, 90.1, 25, 34), (15, 90.7, 96.5, 37, 48), (15, 90.7, 96.0, 35, 46)],
      "WESAD": [(9, 80.1, 89.9, 21, 27), (14, 86.1, 96.4, 34, 42), (14, 84.5, 97.6, 40, 49)],
      "Both": [(19, 79.9, 90.0, 24, 31), (25, 83.5, 93.2, 28, 37), (27, 86.2, 95.0, 32, 41)]}
for s, rows in T2.items():
    r = iv[(s, "normalized")]
    for (w, lab), (at, worst, mean, med, mw) in zip([("window", "window"), ("subject", "person")], rows[:2]):
        st = stats(r, w)
        check(f"{s} {lab} at target", at, st["at"]); check(f"{s} {lab} worst", worst, st["worst"], 1)
        check(f"{s} {lab} mean cov", mean, st["mean"], 1); check(f"{s} {lab} median width", med, st["med"])
        check(f"{s} {lab} mean width", mw, st["meanw"])
    q = pqm[(pqm.setting == s) & (pqm.calibration == "person-quantile")].iloc[0]
    at, worst, mean, med, mw = rows[2]
    check(f"{s} quantile at target", at, q["at target"]); check(f"{s} quantile worst", worst, 100 * q.worst, 1)
    check(f"{s} quantile mean cov", mean, 100 * q["mean cov"], 1); check(f"{s} quantile median width", med, q["median width"])
    check(f"{s} quantile mean width", mw, q["mean width"])
pl = pqm[pqm.calibration == "person-level"].set_index("setting")
for s in T2:
    st = stats(iv[(s, "normalized")], "subject")
    check(f"{s} person-level in quantile script = stored (at)", st["at"], pl.loc[s, "at target"])
check("required level m=14 (%)", 96.4, 100 * 0.9 * (1 + 1 / 14), 1); check("required level m=29 (%)", 93.1, 100 * 0.9 * (1 + 1 / 29), 1)
check("DaLiA quantile narrower by (bpm)", 2, stats(iv[("PPG-DaLiA", "normalized")], "subject")["med"] - pl.loc["PPG-DaLiA", "median width"] + 0
      if False else stats(iv[("PPG-DaLiA", "normalized")], "subject")["med"] - pqm[(pqm.setting == "PPG-DaLiA") & (pqm.calibration == "person-quantile")]["median width"].iloc[0])
check("WESAD quantile wider by (bpm)", 6, pqm[(pqm.setting == "WESAD") & (pqm.calibration == "person-quantile")]["median width"].iloc[0] - stats(iv[("WESAD", "normalized")], "subject")["med"])
for s in ["PPG-DaLiA", "WESAD"]:
    st = stats(iv[(s, "normalized")], "subject")
    check(f"{s} person-level width growth from window (low)", {"PPG-DaLiA": 12, "WESAD": 13}[s], st["med"] - stats(iv[(s, "normalized")], "window")["med"])
rho = {s: spearmanr(iv[(s, "normalized")].hi_subject - iv[(s, "normalized")].lo_subject,
                    (iv[(s, "normalized")].hr_ref - iv[(s, "normalized")].hr_hat).abs())[0] for s in ["PPG-DaLiA", "WESAD"]}
check("Spearman DaLiA", 0.54, rho["PPG-DaLiA"], 2); check("Spearman WESAD", 0.55, rho["WESAD"], 2)
check("Spearman GalaxyPPG", 0.60, spearmanr(ig.hi_subject - ig.lo_subject, (ig.hr_ref - ig.hr_hat).abs())[0], 2)

print("== Ablation of the weighting and correction")
T3 = {"PPG-DaLiA": ([9, 9, 15, 15], [90.1, 90.2, 96.5, 96.5], [25, 25, 37, 37]),
      "WESAD": ([9, 9, 14, 14], [89.9, 90.0, 96.3, 96.4], [21, 21, 33, 34]),
      "Both": ([19, 19, 25, 25], [90.0, 90.0, 93.1, 93.2], [24, 24, 28, 28]),
      "GalaxyPPG (external)": ([14, 14, 21, 21], [91.9, 91.9, 93.9, 94.0], [15, 15, 18, 18])}
order = [("window", "n"), ("person", "n"), ("window", "m"), ("person", "m")]
for s, (ats, covs, wds) in T3.items():
    a = abl[abl.setting == s].set_index(["weighting", "correction"])
    for (w, c), at, cv, wd in zip(order, ats, covs, wds):
        check(f"{s} {w}/{c} at", at, a.loc[(w, c), "at target"]); check(f"{s} {w}/{c} mean cov", cv, 100 * a.loc[(w, c), "coverage (mean over persons)"], 1)
        check(f"{s} {w}/{c} width", wd, a.loc[(w, c), "median width"])
    d = (a.xs("n", level="correction")["median width"].diff().abs().iloc[-1])
    print(f"   {s}: width change from window to person weights with n = {d:.3f} (stated < 0.2)")
b = abl[abl.setting == "BeliefPPG on GalaxyPPG"].set_index(["weighting", "correction"])
check("BeliefPPG window weights + m at target", 19, b.loc[("window", "m"), "at target"])
cov_diff = []
for s in ["PPG-DaLiA", "WESAD", "Both"]:
    for m in ["constant", "normalized", "cqr"]:
        for w in ["window", "subject"]:
            st = stats(iv[(s, m)], w); cov_diff.append(abs(st["pooled"] - st["mean"]))
for w in ["window", "subject"]:
    st = stats(ig, w); cov_diff.append(abs(st["pooled"] - st["mean"]))
for w in ["native", "window", "subject"]:
    st = stats(ib, w); cov_diff.append(abs(st["pooled"] - st["mean"]))
cov_diff += list((100 * (pqm["pooled cov"] - pqm["mean cov"]).abs())) + list(100 * (pqg["pooled cov"] - pqg["mean cov"]).abs())
cov_diff += list(100 * (pqb["pooled cov"] - pqb["mean cov"]).abs())
cov_diff += list(100 * (sg["coverage (all windows)"] - sg["coverage (mean over persons)"]).abs()) if "coverage (all windows)" in sg else []
print(f"   max |pooled - mean over persons| = {max(cov_diff):.3f} points (stated at most 0.4)")

print("== GalaxyPPG results")
T4 = {"constant": [(90.5, 11, 80.4, 49, 48), (91.6, 15, 81.3, 63, 61), (95.5, 23, 92.4, 98, 99)],
      "normalized": [(91.9, 14, 86.1, 15, 25), (94.0, 21, 88.5, 18, 29), (95.3, 23, 90.3, 20, 32)],
      "cqr": [(90.9, 12, 86.0, 12, 22), (94.3, 22, 89.6, 14, 24), (94.9, 23, 90.6, 16, 26)]}
print("   summary_galaxy columns:", list(sg.columns))
for m, rows in T4.items():
    for (wt, lab), (cv, at, worst, med, mw) in zip([("window", "window"), ("subject", "person")], rows[:2]):
        x = sg[(sg.method == m) & (sg.weighting == wt)].iloc[0]
        check(f"{m} {lab} mean cov", cv, 100 * x["coverage (mean over persons)"], 1)
        check(f"{m} {lab} at", at, int(str(x["persons at target"]).split("/")[0]) if "persons at target" in x else x["at target"])
        check(f"{m} {lab} worst", worst, 100 * x["worst person"], 1)
        check(f"{m} {lab} median", med, x["median width (bpm)"]); check(f"{m} {lab} mean width", mw, x["mean width (bpm)"])
    q = pqg[(pqg.score == m) & (pqg.calibration == "person-quantile")].iloc[0]
    cv, at, worst, med, mw = rows[2]
    check(f"{m} quantile mean cov", cv, 100 * q["mean cov"], 1); check(f"{m} quantile at", at, q["at target"])
    check(f"{m} quantile worst", worst, 100 * q.worst, 1); check(f"{m} quantile median", med, q["median width"])
    check(f"{m} quantile mean width", mw, q["mean width"])
    p = pqg[(pqg.score == m) & (pqg.calibration == "person-level")].iloc[0]
    check(f"{m} person-level replay = summary (at)", rows[1][1], p["at target"])
st = stats(ig, "subject"); print(f"   below-target participants: {sorted(100 * st['per'][st['per'] < 0.9])} (stated within 1.5 points of 90)")
check("required level m=30 (%)", 93.0, 100 * 0.9 * (1 + 1 / 30), 1)

print("== Coverage per condition")
def pc(r, cond, w="subject"):
    return 100 * cov(r, w).groupby(cond).mean(), (r[f"hi_{w}"] - r[f"lo_{w}"]).groupby(cond).median()
rd = iv[("PPG-DaLiA", "normalized")]; c, wd = pc(rd, rd.condition)
print("   DaLiA:", {k: round(v) for k, v in c.items()}, "(stated calm 99-100, physical 86-92)")
check("DaLiA sitting width", 19, wd["sitting"]); check("DaLiA stairs width", 77, wd["stairs"])
rw = iv[("WESAD", "normalized")]; c, wd = pc(rw, rw.condition)
print("   WESAD:", {k: round(v, 1) for k, v in c.items()}, "(stated 99-100 calm, stress 83)")
check("WESAD stress coverage", 83, c["stress"])
s11 = rw[rw.subject == "W11"]; c11 = 100 * cov(s11, "subject").groupby(s11.condition).mean()
print("   S11 per condition:", {k: round(v, 1) for k, v in c11.items()}, "(stated 98-100 at rest/amusement/meditation, 13 stress)")
check("S11 stress coverage", 13, c11["stress"])
check("S11 overall person-level", 86.1, 100 * cov(s11, "subject").mean(), 1)
wst, wre = rw[(rw.condition == "stress") & (rw.subject == "W11")], rw[rw.condition == "baseline"]   # S11 stress vs WESAD rest
check("S11 stress median SNR", -5.3, wst.snr_db.median(), 1); check("WESAD rest median SNR", 0.6, wre.snr_db.median(), 1)
check("S11 stress median disagreement", 38, wst.disagree.median()); check("WESAD rest median disagreement", 3, wre.disagree.median())
check("S11 stress median width", 59, (wst.hi_subject - wst.lo_subject).median())
s11st = s11[s11.condition == "stress"]
check("S11 stress median error", 71, (s11st.hr_hat - s11st.hr_ref).abs().median())
galmap = {"baseline": "rest", "rest": "rest", "meditation": "rest", "adaptation": "rest", "screen reading": "rest",
          "standing": "rest", "TSST preparation": "stress", "TSST speech": "stress", "SSST preparation": "stress",
          "SSST singing": "stress", "keyboard typing": "typing", "mobile typing": "typing",
          "walking": "walking", "jogging": "jogging", "running": "running"}
grp = ig.condition.map(galmap); c, wd = pc(ig, grp); err = (ig.hr_hat - ig.hr_ref).abs().groupby(grp).mean()
print("   GalaxyPPG phases:", {k: round(v, 1) for k, v in c.items()})
for ph, cv, e, w in [("walking", 78, 27, 57), ("jogging", 52, 53, 86), ("running", 46, 63, 79)]:
    check(f"Galaxy {ph} coverage", cv, c[ph]); check(f"Galaxy {ph} MAE", e, err[ph]); check(f"Galaxy {ph} width", w, wd[ph])
for ph in ["rest", "typing", "stress"]:
    check(f"Galaxy {ph} coverage", 98, c[ph])
q = pqp[(pqp.setting == "GalaxyPPG") & (pqp.calibration == "person-quantile")].set_index("condition")
check("quantile jogging coverage", 62, 100 * q.loc["jogging", "coverage"]); check("quantile running coverage", 53, 100 * q.loc["running", "coverage"])
keep = ~ig.condition.isin(["jogging", "running"]); st = stats(ig[keep], "subject")
check("excluding jog/run coverage", 96.6, st["pooled"], 1); check("excluding jog/run at target", 23, st["at"])

print("== Split and folds")
sp = pqs.groupby(["setting", "calibration"])
for (s, cal), (mean, lo, hi) in {("Both (LOSO)", "person-level"): (27.1, 26, 28), ("GalaxyPPG", "person-level"): (22.4, 22, 23),
                                 ("Both (LOSO)", "person-quantile"): (27.1, 25, 28)}.items():
    g = sp.get_group((s, cal))["at target"]
    check(f"split {s} {cal} mean at", mean, g.mean(), 1); check(f"split {s} {cal} min", lo, g.min()); check(f"split {s} {cal} max", hi, g.max())
g = sp.get_group(("GalaxyPPG", "person-quantile"))["at target"]
check("split Galaxy quantile partitions with 23/23", 19, (g == 23).sum()); check("split Galaxy quantile other partition", 22, g.min())
cross = {("Both (LOSO)", "person-level"): 28.2617, ("GalaxyPPG", "person-level"): 17.6761,
         ("Both (LOSO)", "person-quantile"): pqm[(pqm.setting == "Both") & (pqm.calibration == "person-quantile")]["median width"].iloc[0],
         ("GalaxyPPG", "person-quantile"): pqg[(pqg.score == "normalized") & (pqg.calibration == "person-quantile")]["median width"].iloc[0]}
diffs = [sp.get_group(k)["median width"].mean() - v for k, v in cross.items()]
print(f"   split minus cross-fit median widths: {[round(d, 2) for d in diffs]} (stated 0.9 to 3.5)")
for s, cal, ats, lo, hi in [("PPG-DaLiA", "person-level", {15}, 36.5, 37.0), ("WESAD", "person-level", {14}, 33.5, 33.8),
                            ("WESAD", "person-quantile", {14}, 38.9, 40.0), ("PPG-DaLiA", "person-quantile", {14, 15}, 34.5, 36.5)]:
    f = pqf[(pqf.setting == s) & (pqf.calibration == cal)]
    print(f"   folds {s} {cal}: at {sorted(f['at target'])}, width {f['median width'].min():.2f}-{f['median width'].max():.2f} (stated {sorted(ats)}, {lo}-{hi})")
f = pqf[(pqf.setting == "PPG-DaLiA") & (pqf.calibration == "person-quantile")]
check("folds DaLiA quantile draws at 14/15", 2, (f["at target"] == 14).sum()); check("folds DaLiA quantile worst", 88.8, 100 * f.worst.min(), 1)

print("== BeliefPPG")
b = pd.read_csv(R + "beliefppg_galaxy.csv"); mg = gal.merge(b, on=["subject", "window"])
check("matched windows", 43001, len(mg))
mb = (mg.hr_belief - mg.hr_ref).abs().groupby(mg.subject).mean(); mo = (mg.hr_hat - mg.hr_ref).abs().groupby(mg.subject).mean()
check("BeliefPPG MAE", 3.2, mb.mean(), 1); check("BeliefPPG SD", 1.5, mb.std(), 1)
check("our MAE same windows", 8.2, mo.mean(), 1); check("our SD same windows", 3.4, mo.std(), 1)
for w, lab, (cv, at, worst, med, mw) in [("native", "Gaussian", (90.6, 15, 76.0, 7, 9)), ("window", "window", (89.9, 15, 74.3, 6, 9)),
                                          ("subject", "person", (94.0, 20, 83.3, 8, 11))]:
    st = stats(ib, w)
    check(f"Belief {lab} mean cov", cv if lab != "Gaussian" else 90.5, st["mean"], 1); check(f"Belief {lab} at", at, st["at"]); check(f"Belief {lab} worst", worst, st["worst"], 1)
    check(f"Belief {lab} median", med, st["med"]); check(f"Belief {lab} mean width", mw, st["meanw"])
q = pqb[pqb.calibration == "person-quantile"].iloc[0]
check("Belief quantile mean cov", 95.7, 100 * q["mean cov"], 1); check("Belief quantile at", 21, q["at target"])
check("Belief quantile worst", 87.9, 100 * q.worst, 1); check("Belief quantile median", 9, q["median width"]); check("Belief quantile mean w", 13, q["mean width"])
run = ib.condition == "running"; check("Belief running person-level", 56, 100 * cov(ib[run], "subject").mean())
check("required level m=22 (%)", 94.1, 100 * 0.9 * (1 + 1 / 22), 1); check("k for m=22", 21, int(np.ceil(round(0.9 * 23, 9))))
d = pd.read_csv(R + "beliefppg_dev_raw.csv"); d["k"] = d.window - d.groupby("subject").window.transform("min")
maes = {}
for o in range(9):
    x = d.assign(window=d.k + o).merge(dev[["subject", "window", "hr_ref"]], on=["subject", "window"])
    maes[o] = (x.hr_belief - x.hr_ref).abs().groupby(x.subject).mean().mean()
check("best offset", 5, min(maes, key=maes.get)); check("MAE offset 5", 2.5, maes[5], 1)
check("MAE offset 4", 2.8, maes[4], 1); check("MAE offset 6", 2.9, maes[6], 1)

print("== Score functions and CQR crossing")
TS = {("PPG-DaLiA", "constant"): (13, 81.0, 96.3, 97, 95), ("PPG-DaLiA", "normalized"): (15, 90.7, 96.5, 37, 48),
      ("PPG-DaLiA", "cqr"): (14, 88.1, 96.5, 37, 45), ("WESAD", "constant"): (14, 84.9, 96.4, 68, 67),
      ("WESAD", "normalized"): (14, 86.1, 96.4, 34, 42), ("WESAD", "cqr"): (14, 87.6, 96.3, 32, 37),
      ("Both", "constant"): (24, 79.2, 93.1, 62, 59), ("Both", "normalized"): (25, 83.5, 93.2, 28, 37),
      ("Both", "cqr"): (24, 86.2, 93.1, 27, 34)}
for (s, m), (at, worst, mean, med, mw) in TS.items():
    st = stats(iv[(s, m)], "subject")
    check(f"S-I {s} {m} at", at, st["at"]); check(f"S-I {s} {m} worst", worst, st["worst"], 1)
    check(f"S-I {s} {m} mean", mean, st["mean"], 1); check(f"S-I {s} {m} median", med, st["med"]); check(f"S-I {s} {m} mean w", mw, st["meanw"])
c = cq.set_index("setting")
check("crossing WESAD", 20, c.loc["WESAD", "crossing windows"]); check("crossing Both", 2, c.loc["DaLiA+WESAD", "crossing windows"])
check("crossing DaLiA", 0, c.loc["DaLiA", "crossing windows"]); check("crossing Galaxy", 0, c.loc["GalaxyPPG", "crossing windows"])
check("empty person-level total", 0, c["empty (subject)"].sum()); check("empty window WESAD", 19, c.loc["WESAD", "empty (window)"])
check("empty window Both", 1, c.loc["DaLiA+WESAD", "empty (window)"])

print("== Calibration rules, bootstrap, curve and rejection")
ru = pd.read_csv(R + "rules_summary.csv").set_index(["setting", "rule"])
for (st, rl), (at, w) in {("GalaxyPPG", "motion-stratified"): (21, 12.6), ("Both", "motion-stratified"): (24, 29.6),
                          ("GalaxyPPG", "CDF pooling"): (14, 15.0), ("GalaxyPPG", "repeated subsampling"): (21, 17.6),
                          ("Both", "CDF pooling"): (19, 23.9), ("Both", "repeated subsampling"): (25, 28.2)}.items():
    check(f"{st} {rl} at", at, ru.loc[(st, rl), "at target"]); check(f"{st} {rl} width", w, ru.loc[(st, rl), "median width"], 1)
check("single subsampling Galaxy min", 6, ru.loc[("GalaxyPPG", "single subsampling"), "at target min"])
check("single subsampling Galaxy max", 23, ru.loc[("GalaxyPPG", "single subsampling"), "at target max"])
check("single subsampling Galaxy mean", 16.1, ru.loc[("GalaxyPPG", "single subsampling"), "at target"], 1)
check("single subsampling Both mean", 17.3, ru.loc[("Both", "single subsampling"), "at target"], 1)
check("single subsampling Both min", 11, ru.loc[("Both", "single subsampling"), "at target min"])
check("single subsampling Both max", 22, ru.loc[("Both", "single subsampling"), "at target max"])
check("single subsampling Both width", 25.1, ru.loc[("Both", "single subsampling"), "median width"], 1)
check("single subsampling Galaxy width", 16.7, ru.loc[("GalaxyPPG", "single subsampling"), "median width"], 1)
cv = pd.read_csv(R + "rules_curve.csv"); g = cv[cv.setting == "GalaxyPPG"]
check("curve Galaxy at 12.6 bpm", 9, g.loc[(g["median width"] - 12.63).abs().idxmin(), "at target"])
bt = pd.read_csv(R + "rules_bootstrap.csv").set_index(["setting", "quantity"])
for (st, qn), (est, lo, hi) in {("PPG-DaLiA+WESAD (separate)", "difference person-level - window-level"): (11, 6, 16),
                                ("Both", "difference person-level - window-level"): (6, 2, 11),
                                ("Both", "difference person-quantile - person-level"): (2, 0, 5),
                                ("GalaxyPPG", "difference person-level - window-level"): (7, 3, 12),
                                ("GalaxyPPG", "difference person-quantile - person-level"): (2, 0, 5)}.items():
    x = bt.loc[(st, qn)]
    check(f"boot {st} {qn} est", est, x.estimate); check(f"boot {st} {qn} lo", lo, x["ci low"]); check(f"boot {st} {qn} hi", hi, x["ci high"])
ac = pd.read_csv(R + "rules_activity.csv").set_index(["setting", "rule", "condition"])
for (st, cd), (a1, a2) in {("GalaxyPPG", "walking"): (78, 86), ("GalaxyPPG", "jogging"): (52, 72), ("GalaxyPPG", "running"): (46, 64),
                           ("WESAD", "stress"): (83, 83), ("PPG-DaLiA", "table soccer"): (86, 86)}.items():
    check(f"act {st} {cd} person", a1, 100 * ac.loc[(st, "person-level", cd), "coverage"])
    check(f"act {st} {cd} motion", a2, 100 * ac.loc[(st, "motion-stratified", cd), "coverage"])
for cd in ["walking", "cycling", "stairs"]:
    check(f"DaLiA {cd} gain (3 points)", 3, 100 * (ac.loc[("PPG-DaLiA", "motion-stratified", cd), "coverage"] - ac.loc[("PPG-DaLiA", "person-level", cd), "coverage"]))
for cd in ["rest", "typing", "stress"]:
    d = 100 * (ac.loc[("GalaxyPPG", "person-level", cd), "coverage"] - ac.loc[("GalaxyPPG", "motion-stratified", cd), "coverage"])
    print(f"   Galaxy {cd} loss {d:.2f} (stated 0 to 2)")
check("rest width person", 16, ac.loc[("GalaxyPPG", "person-level", "rest"), "median width"])
check("rest width motion", 11, ac.loc[("GalaxyPPG", "motion-stratified", "rest"), "median width"])
check("motion width walking", 80, ac.loc[("GalaxyPPG", "motion-stratified", "walking"), "median width"])
check("motion width jogging", 112, ac.loc[("GalaxyPPG", "motion-stratified", "jogging"), "median width"])
ut = pd.read_csv(R + "rules_rejection.csv").set_index(["setting", "retained"])
for (st, f), v in {("GalaxyPPG", 1.0): 8.3, ("GalaxyPPG", 0.9): 4.3, ("GalaxyPPG", 0.8): 3.0, ("Both", 1.0): 8.9, ("Both", 0.9): 6.2,
                   ("Both", 0.8): 4.9, ("BeliefPPG (GalaxyPPG)", 1.0): 3.2, ("BeliefPPG (GalaxyPPG)", 0.9): 1.9}.items():
    check(f"utility {st} {f}", v, ut.loc[(st, f), "MAE retained"], 1)
se = pd.read_csv(R + "galaxy_sensitivity.csv").set_index(["subset", "rule"])
for rl, v in [("window-level", 10), ("person-level", 14), ("person-quantile", 16)]:
    check(f"no clock-shift {rl}", v, se.loc[("without clock-shifted participants", rl), "at target"])
    check(f"no <40 {rl} unchanged", se.loc[("all", rl), "at target"], se.loc[("without references below 40 bpm", rl), "at target"])
e = np.quantile(dev.motion, [1 / 3, 2 / 3]); sd = 10 ** e - 1e-3
check("tercile edge 1 (g)", 0.012, sd[0], 3); check("tercile edge 2 (g)", 0.10, sd[1], 2)
check("k m=30", 28, int(np.ceil(round(0.9 * 31, 9)))); check("k m=19", 18, int(np.ceil(round(0.9 * 20, 9)))); check("k m=20", 19, int(np.ceil(round(0.9 * 21, 9))))
check("level m=19", 94.7, 100 * 0.9 * (1 + 1 / 19), 1); check("level m=20", 94.5, 100 * 0.9 * (1 + 1 / 20), 1)
gw = pd.read_csv(R + "rules_windows.csv"); g12 = gw[(gw.setting == "GalaxyPPG") & (gw.subject == "G12") & (gw.condition == "running")]
gd = gal[(gal.subject == "G12") & (gal.condition == "running")]
check("G12 running disagreement", 60, gd.disagree.median()); check("G12 running SNR", -6.7, gd.snr_db.median(), 1)
check("G12 running width", 176, (2 * g12["q_person-level"] * g12.sigma).median()); check("G12 running error", 98, (g12.hr_hat - g12.hr_ref).abs().median())
sp2 = pqs.groupby(["setting", "calibration"])["at target"].mean()
check("split Galaxy quantile mean", 22.95, sp2[("GalaxyPPG", "person-quantile")], 2)

rv = pd.read_csv(R + "rpeak_validation_dalia.csv"); kept = rv.windows * rv["kept by QC (%)"] / 100
check("R-peak kept (%)", 98.2, 100 * kept.sum() / rv.windows.sum(), 1)

check("R-peak MAE", 0.46, (rv["MAE (bpm)"] * kept).sum() / kept.sum(), 2)

# motion-stratified calibration (exploratory)
a2 = pd.read_csv(R + "stratified_bootstrap.csv").set_index(["setting", "quantity"])
d = "difference motion-stratified - person-level"
check("MS-PL dev", -1, a2.loc[("Both", d), "estimate"]); check("MS-PL dev lo", -5, a2.loc[("Both", d), "ci low"]); check("MS-PL dev hi", 3, a2.loc[("Both", d), "ci high"])
check("MS-PL gal", 0, a2.loc[("GalaxyPPG", d), "estimate"]); check("MS-PL gal lo", -3, a2.loc[("GalaxyPPG", d), "ci low"]); check("MS-PL gal hi", 3, a2.loc[("GalaxyPPG", d), "ci high"])
ab = pd.read_csv(R + "stratified_activity_bootstrap.csv").set_index(["setting", "activity"])
for act, v in [("walking", (9, 4, 14)), ("jogging", (20, 12, 27)), ("running", (19, 12, 26))]:
    r = ab.loc[("GalaxyPPG", act)]
    check(f"MS gain {act}", v[0], 100 * r["diff"]); check(f"MS gain {act} lo", v[1], 100 * r["diff lo"]); check(f"MS gain {act} hi", v[2], 100 * r["diff hi"])
wc = pd.read_csv(R + "stratified_width_by_class.csv"); g = wc[wc.setting == "GalaxyPPG"].set_index("motion class")
check("calm share", 63, 100 * g.loc["0", "share of windows"]); check("calm width PL", 15.4, g.loc["0", "person-level median width"], 1)
check("calm width MS", 10.6, g.loc["0", "motion-stratified median width"], 1)
check("mean width MS", 28.1, g.loc["all", "motion-stratified mean width"], 1); check("mean width PL", 28.7, g.loc["all", "person-level mean width"], 1)
rr = pd.read_csv(R + "rules_summary.csv").set_index(["setting", "rule"])
check("dev MS width", 29.6, rr.loc[("Both", "motion-stratified"), "median width"], 1); check("dev PL width", 28.3, rr.loc[("Both", "person-level"), "median width"], 1)
check("gal MS width", 12.6, rr.loc[("GalaxyPPG", "motion-stratified"), "median width"], 1)
bs = pd.read_csv(R + "stratified_belief.csv").set_index("rule")
check("Belief MS run", 82, 100 * bs.loc["motion-stratified", "running cov"]); check("Belief PL run", 56, 100 * bs.loc["person-level", "running cov"])
check("Belief PL jog", 75, 100 * bs.loc["person-level", "jogging cov"]); check("Belief MS jog", 90, 100 * bs.loc["motion-stratified", "jogging cov"])
check("Belief MS at target", 18, bs.loc["motion-stratified", "at target"]); check("Belief PL mean width", 10.9, bs.loc["person-level", "mean width"], 1)
check("Belief MS mean width", 16.8, bs.loc["motion-stratified", "mean width"], 1)
check("Belief MS run width", 63, bs.loc["motion-stratified", "running width"]); check("Belief MS jog width", 50, bs.loc["motion-stratified", "jogging width"])
check("Belief PL run width", 18, bs.loc["person-level", "running width"]); check("Belief PL jog width", 14, bs.loc["person-level", "jogging width"])
check("Belief MS mean cov", 94.1, 100 * bs.loc["motion-stratified", "mean cov"], 1); check("Belief MS worst", 81.9, 100 * bs.loc["motion-stratified", "worst"], 1)
check("Belief MS median width", 7, bs.loc["motion-stratified", "median width"]); check("Belief MS mean width r", 17, bs.loc["motion-stratified", "mean width"])
check("MS jog width", 112, rr.loc[("GalaxyPPG", "motion-stratified"), "median width"] * 0 + pd.read_csv(R + "stratified_activity_bootstrap.csv").set_index(["setting", "activity"]).loc[("GalaxyPPG", "jogging"), "motion-stratified median width"])
abx = pd.read_csv(R + "stratified_activity_bootstrap.csv").set_index(["setting", "activity"])
check("MS run width", 103, abx.loc[("GalaxyPPG", "running"), "motion-stratified median width"]); check("MS walk width", 80, abx.loc[("GalaxyPPG", "walking"), "motion-stratified median width"])
dg = [100 * abx.loc[("Both", a), "diff"] for a in ["walking", "cycling", "table soccer", "stairs"]]
check("dev pooled gain min", 7, min(dg)); check("dev pooled gain max", 14, max(dg))
check("dev pooled meditation", -6, 100 * abx.loc[("Both", "meditation"), "diff"]); check("dev pooled stress", -4, 100 * abx.loc[("Both", "stress"), "diff"])
check("dev pooled amusement", -4, 100 * abx.loc[("Both", "amusement"), "diff"])
tt = pd.read_csv(R + "tradeoff_table.csv").set_index(["setting", "rule"])
for (st, ru), (w, g) in {("Both", "window-level"): (23.8, 23.9), ("Both", "person-level"): (28.3, 28.3), ("Both", "person-quantile"): (32.2, 30.9),
                         ("Both", "motion-stratified"): (29.6, 28.3), ("GalaxyPPG", "window-level"): (15.0, 15.1), ("GalaxyPPG", "person-level"): (17.7, 17.3),
                         ("GalaxyPPG", "person-quantile"): (20.1, 20.1), ("GalaxyPPG", "motion-stratified"): (12.6, 17.3)}.items():
    check(f"tradeoff width {st} {ru}", w, tt.loc[(st, ru), "median width"], 1); check(f"tradeoff global {st} {ru}", g, tt.loc[(st, ru), "global threshold width"], 1)

print("== PPG-Sport and cross-sensor calibration")
from pathlib import Path as _P
if _P("results/motionrich_summary.csv").exists():
    ms = pd.read_csv("results/motionrich_summary.csv").set_index(["setting", "calibration"])
    ma = pd.read_csv("results/motionrich_activities.csv").set_index(["setting", "calibration", "activity"])
    mb = pd.read_csv("results/motionrich_bootstrap.csv")
    mm = pd.read_csv("results/motionrich_model.csv").set_index("setting")
    D, L, R = "PPG-DaLiA + WESAD", "+ PPG-Sport (left)", "+ PPG-Sport (right)"
    rows = {(D, "window-level"): (14, 15.0, 71, 38, 38), (D, "person-level"): (21, 17.7, 78, 52, 46),
            (D, "person-quantile"): (23, 20.1, 82, 62, 53), (D, "same width"): (23, 20.7, 83, 64, 55),
            (L, "window-level"): (23, 20.7, 88, 77, 69), (L, "person-level"): (23, 32.0, 92, 81, 74),
            (L, "person-quantile"): (23, 41.6, 94, 87, 77), (L, "motion-stratified"): (22, 11.7, 93, 84, 76)}
    for (st, ru), (n, w, wa, jo, ru_) in rows.items():
        check(f"T4 at target {st} {ru}", n, ms.loc[(st, ru), "at target"])
        check(f"T4 width {st} {ru}", w, ms.loc[(st, ru), "median width"], 1)
        for a, v in [("walking", wa), ("jogging", jo), ("running", ru_)]:
            check(f"T4 {a} {st} {ru}", v, 100 * ma.loc[(st, ru, a), "coverage"])
    check("right wrist width", 19.3, ms.loc[(R, "window-level"), "median width"], 1)
    check("right wrist running", 72, 100 * ma.loc[(R, "window-level", "running"), "coverage"])
    check("jogging width", 139, ma.loc[(L, "window-level", "jogging"), "median width"])
    check("running width", 95, ma.loc[(L, "window-level", "running"), "median width"])
    check("coef motion dev", 0.12, mm.loc[D, "coef x SD motion"], 2); check("coef motion +sport", 0.32, mm.loc[L, "coef x SD motion"], 2)
    check("coef disagree +sport", 0.58, mm.loc[L, "coef x SD disagree"], 2); check("coef snr +sport", 0.04, mm.loc[L, "coef x SD snr"], 2)
    def bt(a, b, act):
        r = mb[(mb.a == a) & (mb.b == b) & (mb.activity == act)].iloc[0]; return r["difference (points)"], r["ci low"], r["ci high"]
    for act, vals in [("walking", (17, 8, 27)), ("jogging", (38, 27, 49)), ("running", (30, 19, 41))]:
        for lab, st, v in zip(["est", "lo", "hi"], vals, bt(f"{L} / window-level", f"{D} / window-level", act)):
            check(f"boot window {act} {lab}", st, v)
    for act, vals in [("jogging", (13, 7, 20)), ("running", (14, 7, 22))]:
        for lab, st, v in zip(["est", "lo", "hi"], vals, bt(f"{L} / window-level", f"{D} / same width", act)):
            check(f"boot same width {act} {lab}", st, v)
    for lab, st, v in zip(["est", "lo", "hi"], (0.0, -0.4, 0.5), bt(f"{L} / window-level", f"{D} / same width", "all")):
        check(f"boot same width all {lab}", st, v, 1)
    for w_, mae_st in [("L", (25.5, 11.5)), ("R", (37.7, 15.4))]:
        t = pd.read_csv(f"results/windows_ppgsport_{w_}.csv"); pm = (t.hr_hat - t.hr_ref).abs().groupby(t.subject).mean()
        check(f"PPG-Sport MAE {w_}", mae_st[0], pm.mean(), 1); check(f"PPG-Sport MAE SD {w_}", mae_st[1], pm.std(), 1)
        sm = pd.read_csv(f"results/summary_ppgsport_{w_}.csv")
        check(f"PPG-Sport max at target {w_}", 2 if w_ == "L" else 1, sm["at target"].max())
    t = pd.read_csv("results/windows_ppgsport_L.csv"); ae = (t.hr_hat - t.hr_ref).abs()
    for a, v in [("stationary", 2.7), ("badminton", 13.2), ("running", 15.9), ("tabletennis", 21.1), ("walking", 34.5), ("basketball", 65.4)]:
        check(f"PPG-Sport MAE {a}", v, ae[t.condition == a].mean(), 1)

print("== Robustness of the cross-sensor result and E4 control")
if _P("results/e4_control.csv").exists():
    o = pd.read_csv("results/e4_control.csv")
    g = o[(o.setting == "galaxy only") & (o.rule == "window")]
    check("E4 control jogging", 69, 100 * g[g.act == "jogging"].cv.mean()); check("E4 control running", 57, 100 * g[g.act == "running"].cv.mean())
    check("E4 control width", 10.4, g.w.median(), 1); check("E4 control at target", 11, (g.groupby("subject").cv.mean() >= 0.9).sum())
    p_ = o[(o.setting == "dev + galaxy") & (o.rule == "window")]
    check("E4 pooled running", 43, 100 * p_[p_.act == "running"].cv.mean())
    check("same width mean", 33.0, ms.loc[(D, "same width"), "mean width"], 1); check("cross-sensor mean width", 34.1, ms.loc[(L, "window-level"), "mean width"], 1)

print("== Nested selection of the MAST settings")
if _P("results/mast_nested.csv").exists():
    nn = pd.read_csv("results/mast_nested.csv"); nn["ds"] = nn.subject.str[0]
    check("nested MAE DaLiA", 9.2, nn[nn.ds == "D"].nested.mean(), 1); check("nested MAE WESAD", 7.2, nn[nn.ds == "W"].nested.mean(), 1)
    check("chosen MAE DaLiA", 9.9, nn[nn.ds == "D"].chosen.mean(), 1); check("chosen MAE WESAD", 7.3, nn[nn.ds == "W"].chosen.mean(), 1)

print("== Causal check (test_causal_vs_zero_phase.py)")
if _P("results/causal_check.csv").exists():
    cc = pd.read_csv("results/causal_check.csv")
    if "variant" not in cc.columns: cc = cc.rename(columns={"mode": "variant"})
    m_ = cc.groupby(["variant", "dataset"]).mae.mean()
    check("causal MAE DaLiA", 10.0, m_[("causal", "DaLiA")], 1); check("causal MAE WESAD", 7.6, m_[("causal", "WESAD")], 1)
print("Not recomputed by this script: the WESAD ECG quality control (step01b) and the BeliefPPG window alignment (beliefppg/run_beliefppg.py).")

print(f"\n{ok} OK, {bad} MISMATCH")
