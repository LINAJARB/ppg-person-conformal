"""Step 1: estimated HR and descriptors for all development subjects."""
import sys, time
from pathlib import Path
import pandas as pd
from ppgs.data import load_dalia, load_wesad, DALIA_IDS, WESAD_IDS
from ppgs.estimator import process_record

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("compact")
out = Path("results"); out.mkdir(exist_ok=True)
parts, qc = [], []
for loader, ids in [(load_dalia, DALIA_IDS), (load_wesad, WESAD_IDS)]:
    for sid in ids:
        t0 = time.time()
        rec = loader(folder, sid)
        df = process_record(rec)
        qc.append({"dataset": df.dataset[0], "windows with a reference": len(df), "windows": len(rec["hr_ref"])})
        mae = (df.hr_hat - df.hr_ref).abs().mean()
        print(f"{df.subject[0]:>4s} {len(df):5d} windows  MAE {mae:5.1f} bpm  ({time.time()-t0:.0f} s)", flush=True)
        parts.append(df)
df = pd.concat(parts, ignore_index=True)
df.to_csv(out / "windows.csv", index=False)
print(df.groupby("dataset").apply(lambda g: (g.hr_hat - g.hr_ref).abs().mean()))
qc = pd.DataFrame(qc).groupby("dataset").sum()
qc["excluded (%)"] = 100 * (1 - qc["windows with a reference"] / qc["windows"])
qc.to_csv(out / "reference_quality_control.csv")
print(qc)
