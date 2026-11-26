"""Step 15c: optimism of the MAST settings. The four settings are varied on a
3x3x3x3 grid (half, chosen, double). Nested leave-one-subject-out: for each
development participant, the setting with the lowest mean per-participant MAE
on the other 29 is selected and applied to the held-out participant.
Usage: python step15c_mast_nested.py <compact folder>. Output: results/mast_nested.csv"""
import sys, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.ndimage import maximum_filter1d
import ppgs.estimator as E
from ppgs.data import load_dalia, load_wesad, DALIA_IDS, WESAD_IDS, FS_BVP, FS_ACC, WIN_S, SHIFT_S

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("compact")
GRID = E.GRID
pre = {}
for loader, ids in [(load_dalia, DALIA_IDS), (load_wesad, WESAD_IDS)]:
    for sid in ids:
        rec = loader(folder, sid)
        bvp, acc = E.bandpass(rec["bvp"], FS_BVP), E.bandpass(rec["acc"], FS_ACC)
        mag = np.linalg.norm(rec["acc"], axis=1)
        fp, fa = np.fft.rfftfreq(E.NFFT, 1 / FS_BVP), np.fft.rfftfreq(E.NFFT, 1 / FS_ACC)
        n = min(len(rec["hr_ref"]), (len(bvp) - WIN_S * FS_BVP) // (SHIFT_S * FS_BVP) + 1, (len(acc) - WIN_S * FS_ACC) // (SHIFT_S * FS_ACC) + 1)
        P, M, v = np.empty((n, len(GRID))), np.empty((n, len(GRID))), np.empty(n)
        for i in range(n):
            x = bvp[i * SHIFT_S * FS_BVP: i * SHIFT_S * FS_BVP + WIN_S * FS_BVP]
            a = acc[i * SHIFT_S * FS_ACC: i * SHIFT_S * FS_ACC + WIN_S * FS_ACC]
            P[i] = E.pool_on_grid(E.power_spectrum(x), fp)
            pa = maximum_filter1d(E.pool_on_grid(E.power_spectrum(a), fa), 2 * E.MASK_WIDTH + 1)
            M[i] = pa / max(pa.max(), 1e-12)
            v[i] = np.std(mag[i * SHIFT_S * FS_ACC: i * SHIFT_S * FS_ACC + WIN_S * FS_ACC])
        pre[rec["subject"]] = (P, M, v, rec["hr_ref"][:n])
        print(rec["subject"], n, flush=True)

grid = list(itertools.product([0.025, 0.05, 0.1], [1.5, 3.0, 6.0], [6.0, 12.0, 24.0], [0.25, 0.5, 1.0]))
rows = []
for mref, sd, jump, obs in grid:
    E.STEP_SD, E.JUMP, E.OBS_POWER = sd, jump, obs
    for s, (P, M, v, ref) in pre.items():
        lam = np.minimum(1.0, v / mref)[:, None]
        hat = E.online_viterbi(P * np.clip(1 - lam * M, 0, 1) ** 2)
        ok = np.isfinite(ref)
        rows.append({"motion_ref": mref, "step_sd": sd, "jump": jump, "obs_power": obs, "subject": s,
                     "mae": np.abs(hat[ok] - ref[ok]).mean()})
    print(mref, sd, jump, obs, flush=True)
r = pd.DataFrame(rows); r.to_csv("results/mast_grid.csv", index=False)
keys = ["motion_ref", "step_sd", "jump", "obs_power"]
chosen = (0.05, 3.0, 12.0, 0.5)
piv = r.pivot_table(index=keys, columns="subject", values="mae")
out = []
for s in piv.columns:
    best = piv.drop(columns=s).mean(axis=1).idxmin()
    out.append({"subject": s, "nested": piv.loc[best, s], "chosen": piv.loc[chosen, s], "selected": best})
o = pd.DataFrame(out); o.to_csv("results/mast_nested.csv", index=False)
o["dataset"] = o.subject.str[0]
print(o.groupby("dataset")[["nested", "chosen"]].mean().round(2))
print("selected = chosen for", sum(o.selected == chosen), "of", len(o))
print("grid MAE range over settings:", piv.mean(axis=1).min().round(2), piv.mean(axis=1).max().round(2))
