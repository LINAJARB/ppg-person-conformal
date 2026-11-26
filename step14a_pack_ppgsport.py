"""Step 14a (optional): pack the PPG-Sport repository into one .npz per activity and wrist.

Usage: python step14a_pack_ppgsport.py <PPG-Sport folder containing dataset/ and cleaned_gt/> [output folder]
The packs are read by step14_ppgsport_external.py.
"""
import os
import sys
import numpy as np

base = sys.argv[1] if len(sys.argv) > 1 else os.path.join("PPG-Sport-main")
out = sys.argv[2] if len(sys.argv) > 2 else "ppgsport_pack"
os.makedirs(out, exist_ok=True)
for act in ["stationary", "walking", "running", "badminton", "tabletennis", "basketball"]:
    for w in "LR":
        d = {}
        for i in range(1, 31):
            s = "S%02d" % i
            f = os.path.join(base, "cleaned_gt", "%s_%s_hr.npy" % (s, act))
            if os.path.exists(f):
                d[s + "_hr"] = np.load(f).astype(np.float32)
            for k in ["ppg", "imu"]:
                f = os.path.join(base, "dataset", "%s%s_%s_%s.npy" % (s, w, act, k))
                if os.path.exists(f):
                    d["%s%s_%s" % (s, w, k)] = np.load(f).astype(np.float32)
        np.savez_compressed(os.path.join(out, "%s_%s.npz" % (act, w)), **d)
        print(act, w, len(d), "arrays")
