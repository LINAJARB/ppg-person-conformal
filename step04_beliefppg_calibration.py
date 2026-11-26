"""Second estimator: BeliefPPG on GalaxyPPG, person-level calibration within GalaxyPPG.

BeliefPPG was developed on datasets that include PPG-DaLiA and WESAD,
so only GalaxyPPG (published after the weights) is used. Each participant is
calibrated on the 22 other participants (leave-one-subject-out).
"""
import numpy as np
import pandas as pd
import ppgs.conformal as C
from ppgs.conformal import loso, summarize

b = pd.read_csv("results/beliefppg_galaxy.csv")
g = pd.read_csv("results/windows_galaxy.csv")
df = g.merge(b, on=["subject", "window"], how="inner").rename(columns={"hr_hat": "hr_ours"})
df["hr_hat"] = df.hr_belief
df["log_unc"] = np.log(df.unc_belief)
df["dataset"] = "GalaxyPPG"
mae_b = (df.hr_belief - df.hr_ref).abs().groupby(df.subject).mean()
mae_o = (df.hr_ours - df.hr_ref).abs().groupby(df.subject).mean()
print(f"GalaxyPPG windows without a BeliefPPG output: {len(g) - len(df)}")
print(f"windows {len(df)}  MAE BeliefPPG {mae_b.mean():.2f} +/- {mae_b.std():.2f}  ours {mae_o.mean():.2f} +/- {mae_o.std():.2f}")

rows, phases = [], []
def add(name, res, w):
    rows.append({"intervals": name, "weighting": w, **summarize(res, w)})
    cov = (res.hr_ref >= res[f"lo_{w}"]) & (res.hr_ref <= res[f"hi_{w}"])
    wid = res[f"hi_{w}"] - res[f"lo_{w}"]
    for ph in ["baseline", "TSST speech", "walking", "jogging", "running"]:
        s = res.condition == ph
        phases.append({"intervals": name, "weighting": w, "phase": ph, "coverage": cov[s].mean(), "median width": wid[s].median()})

# BeliefPPG's own uncertainty, read as a Gaussian standard deviation (no calibration)
nat = df.copy()
nat["lo_native"] = np.maximum(nat.hr_belief - 1.645 * nat.unc_belief, 40)
nat["hi_native"] = np.minimum(nat.hr_belief + 1.645 * nat.unc_belief, 200)
add("BeliefPPG own uncertainty (+/-1.645 sd)", nat, "native")

for name, method, feats in [("constant", "constant", ["log_unc"]),
                            ("normalized, scale from BeliefPPG uncertainty", "normalized", ["log_unc"]),
                            ("normalized, our three descriptors", "normalized", ["snr_db", "motion", "disagree"])]:
    C.FEATURES[:] = feats
    res = loso(df, method)
    for w in ("window", "subject"):
        add(name, res, w)
    if name.startswith("normalized, scale from BeliefPPG"):
        res["lo_native"], res["hi_native"] = nat.lo_native.to_numpy(), nat.hi_native.to_numpy()
        res.to_csv("results/intervals_galaxy_beliefppg.csv", index=False)
tab = pd.DataFrame(rows); ph = pd.DataFrame(phases)
tab.to_csv("results/summary_beliefppg_galaxy.csv", index=False); ph.to_csv("results/summary_beliefppg_galaxy_phases.csv", index=False)
pd.set_option("display.width", 250)
print(tab.round(3).to_string(index=False))
print(ph[ph.weighting.isin(["native", "subject"])].round(3).to_string(index=False))
