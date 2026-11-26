"""Step 3 (run once): final test on GalaxyPPG, calibrated on DaLiA + WESAD.

Usage: python step03_galaxyppg_external.py <path to GalaxyPPG/Dataset>
Needs results/windows.csv from step01b_development_windows.py (development data).
All settings were fixed on the development data before this test.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from ppgs.estimator import process_record
from ppgs.galaxy import list_galaxy, load_galaxy
from ppgs.conformal import calibrate_and_apply, summarize, METHODS

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../GalaxyPPG/Dataset")
out = Path("results"); out.mkdir(exist_ok=True)

parts, info = [], []
for sid in list_galaxy(root):
    t0 = time.time()
    try:
        rec = load_galaxy(root, sid)
    except Exception as err:  # report and skip a participant whose files cannot be read
        print(f"P{sid:02d} skipped: {err}", flush=True)
        continue
    df = process_record(rec)
    mae = (df.hr_hat - df.hr_ref).abs().mean()
    off, corr, applied = rec["sync"]
    info.append({"participant": f"P{sid:02d}", "windows": len(df), "MAE (bpm)": mae,
                 "chest clock offset (s)": off, "accelerometer correlation": corr, "offset applied": applied})
    print(f"P{sid:02d} {len(df):5d} windows  MAE {mae:5.1f} bpm  offset {off:+.1f} s "
          f"({'applied' if applied else 'not applied'})  ({time.time() - t0:.0f} s)", flush=True)
    parts.append(df)
test = pd.concat(parts, ignore_index=True)
test.to_csv(out / "windows_galaxy.csv", index=False)
pd.DataFrame(info).to_csv(out / "galaxy_participants.csv", index=False)

dev = pd.read_csv(out / "windows.csv")
rows, phase_rows = [], []
for m in METHODS:
    res = calibrate_and_apply(dev, test, m)
    res.to_csv(out / f"intervals_galaxy_{m}.csv", index=False)
    for w in ("window", "subject"):
        rows.append({"method": m, "weighting": w, **summarize(res, w)})
        cov = (res.hr_ref >= res[f"lo_{w}"]) & (res.hr_ref <= res[f"hi_{w}"])
        for ph, g in cov.groupby(res.condition):
            phase_rows.append({"method": m, "weighting": w, "phase": ph, "coverage": g.mean(), "windows": len(g)})
tab = pd.DataFrame(rows)
tab.to_csv(out / "summary_galaxy.csv", index=False)
pd.DataFrame(phase_rows).to_csv(out / "summary_galaxy_phases.csv", index=False)
persons = test.groupby("subject").apply(lambda g: (g.hr_hat - g.hr_ref).abs().mean())
print(f"\nGalaxyPPG: {test.subject.nunique()} participants, {len(test)} windows, "
      f"MAE {persons.mean():.2f} +/- {persons.std():.2f} bpm (mean over participants)")
pd.set_option("display.width", 250)
print(tab.round(3).to_string(index=False))
