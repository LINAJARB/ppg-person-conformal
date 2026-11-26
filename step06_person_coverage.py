"""Per-person coverage of person-quantile calibration (normalized score) for Figure 1.
Reuses the functions of step05_person_quantile.py. Checks that person-level
coverages match the stored interval files."""
import numpy as np
import pandas as pd
src = open("step05_person_quantile.py").read().split("# 1. leave-one-subject-out")[0]
exec(src)

def ext(dev, test, feats):
    X, e = dev[feats].to_numpy(), (dev.hr_ref - dev.hr_hat).to_numpy()
    subj = dev.subject.to_numpy()
    oof, model = oof_model(X, e, subj, dev.dataset.to_numpy())
    s = model.sigma(test[feats].to_numpy())
    return {k: apply(test, -qf(oof, subj) * s, qf(oof, subj) * s) for k, qf in QUANTILES.items()}

bel = gal.merge(pd.read_csv("results/beliefppg_galaxy.csv"), on=["subject", "window"])
bel["hr_hat"] = bel.hr_belief; bel["log_unc"] = np.log(bel.unc_belief)
runs = {"PPG-DaLiA": internal(dev[dev.dataset == "DaLiA"].reset_index(drop=True), F3),
        "WESAD": internal(dev[dev.dataset == "WESAD"].reset_index(drop=True), F3),
        "GalaxyPPG": ext(dev, gal, F3),
        "BeliefPPG": internal(bel, ["log_unc"])}
stored = {"PPG-DaLiA": "intervals_DaLiA_normalized.csv", "WESAD": "intervals_WESAD_normalized.csv",
          "GalaxyPPG": "intervals_galaxy_normalized.csv", "BeliefPPG": "intervals_galaxy_beliefppg.csv"}
rows = []
for name, res in runs.items():
    r = pd.read_csv("results/" + stored[name])
    ref = ((r.hr_ref >= r.lo_subject) & (r.hr_ref <= r.hi_subject)).groupby(r.subject).mean()
    mine = res["person-level"].groupby("subject")["cov"].mean()
    print(name, "max |person-level - stored| =", float((mine - ref.reindex(mine.index)).abs().max()))
    for k, v in res.items():
        for s, c in v.groupby("subject")["cov"].mean().items():
            rows.append({"setting": name, "calibration": k, "subject": s, "coverage": c})
pd.DataFrame(rows).to_csv("results/person_coverage_by_calibration.csv", index=False)
