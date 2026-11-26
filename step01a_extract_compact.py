"""Extract a compact copy of PPG-DaLiA and WESAD (wrist signals + reference).

Usage, from the repository root:
    python step01a_extract_compact.py <folder that contains PPG_FieldStudy and WESAD>

Writes one small .npz file per subject in the folder "compact":
    compact/dalia_S1.npz ... dalia_S15.npz   (bvp, acc, hr_ref, activity)
    compact/wesad_S2.npz ... wesad_S17.npz   (bvp, acc, ecg, label)
"""

import pickle
from pathlib import Path

import numpy as np

import sys
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
OUT = Path("compact")
OUT.mkdir(exist_ok=True)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


# PPG-DaLiA
for sid in range(1, 16):
    path = ROOT / "PPG_FieldStudy" / f"S{sid}" / f"S{sid}.pkl"
    if not path.exists():
        print("missing", path)
        continue
    d = load(path)
    np.savez_compressed(
        OUT / f"dalia_S{sid}.npz",
        bvp=np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float32).ravel(),
        acc=np.asarray(d["signal"]["wrist"]["ACC"], dtype=np.float32),
        hr_ref=np.asarray(d["label"], dtype=np.float32).ravel(),
        activity=np.asarray(d["activity"]).ravel().astype(np.int8),
    )
    print("done", path.name)
    del d

# WESAD (ECG kept at 700 Hz, label downsampled to 4 Hz)
for sid in range(2, 18):
    path = ROOT / "WESAD" / f"S{sid}" / f"S{sid}.pkl"
    if not path.exists():
        continue
    d = load(path)
    np.savez_compressed(
        OUT / f"wesad_S{sid}.npz",
        bvp=np.asarray(d["signal"]["wrist"]["BVP"], dtype=np.float32).ravel(),
        acc=np.asarray(d["signal"]["wrist"]["ACC"], dtype=np.float32),
        ecg=np.asarray(d["signal"]["chest"]["ECG"], dtype=np.float32).ravel(),
        label=np.asarray(d["label"]).ravel()[::175].astype(np.int8),
    )
    print("done", path.name)
    del d

print("Finished. Files written to", OUT)
