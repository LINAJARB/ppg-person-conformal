"""BeliefPPG heart rate on the three datasets (complementary estimator).

Run from the repository root, in the separate environment of beliefppg/requirements.txt:
    python beliefppg/run_beliefppg.py <compact folder> <GalaxyPPG/Dataset folder>

Writes
  results/beliefppg_dev_raw.csv   PPG-DaLiA and WESAD, raw output index (used only to
                                  choose the alignment offset)
  results/beliefppg_galaxy.csv    GalaxyPPG, aligned on the 8 s / 2 s window grid
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from beliefppg import infer_hr_uncertainty

from ppgs.data import load_dalia, load_wesad, DALIA_IDS, WESAD_IDS
from ppgs.galaxy import list_galaxy, load_galaxy

COMPACT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "compact"
GALAXY = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / "GalaxyPPG" / "Dataset"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
OFFSET = 5  # output j is matched to window j + 5, chosen on PPG-DaLiA and WESAD (step13 checks it)


def run(rec):
    hr, unc, _ = infer_hr_uncertainty(
        np.asarray(rec["bvp"], dtype=float).reshape(-1, 1), 64,
        acc=np.asarray(rec["acc"], dtype=float), acc_freq=32,
        decoding="sumproduct", use_time_backbone=True, uncertainty="std")
    return pd.DataFrame({"subject": rec["subject"], "j": np.arange(len(np.ravel(hr))),
                         "hr_belief": np.ravel(hr), "unc_belief": np.ravel(unc)})


dev = []
for loader, ids in [(load_dalia, DALIA_IDS), (load_wesad, WESAD_IDS)]:
    for sid in ids:
        t0 = time.time()
        dev.append(run(loader(COMPACT, sid)))
        print(f"{dev[-1].subject[0]:>4s} done ({time.time() - t0:.0f} s)", flush=True)
pd.concat(dev).rename(columns={"j": "window"}).to_csv(OUT / "beliefppg_dev_raw.csv", index=False)

gal = []
for sid in list_galaxy(GALAXY):
    t0 = time.time()
    out = run(load_galaxy(GALAXY, sid))
    out["window"] = out.pop("j") + OFFSET
    gal.append(out[["subject", "window", "hr_belief", "unc_belief"]])
    print(f"   G{sid} done ({time.time() - t0:.0f} s)", flush=True)
pd.concat(gal).to_csv(OUT / "beliefppg_galaxy.csv", index=False)
