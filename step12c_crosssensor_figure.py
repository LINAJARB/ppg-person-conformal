"""Figure: exploratory cross-sensor calibration on GalaxyPPG, at equal median width."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path("figures"); OUT.mkdir(exist_ok=True)
BLUE, INK, MUTED, GREY = "#2a78d6", "#1f1f1d", "#6b6a63", "#a8a79e"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
a = pd.read_csv("results/motionrich_activities.csv")
order = ["rest", "stress", "typing", "walking", "jogging", "running"]
A = a[(a.setting == "PPG-DaLiA + WESAD") & (a.calibration == "same width")].set_index("activity").reindex(order)
B = a[(a.setting == "+ PPG-Sport (left)") & (a.calibration == "window-level")].set_index("activity").reindex(order)
y = np.arange(len(order))[::-1]
plt.figure(figsize=(7.16, 2.3))
for k, (col, scale, lab, xmax, ticks) in enumerate([("coverage", 100, "Coverage (%)", 116, [0, 50, 90]),
                                                     ("median width", 1, "Median interval width (bpm)", 175, [0, 50, 100, 150])]):
    plt.subplot(1, 2, k + 1)
    va, vb = A[col] * scale, B[col] * scale
    plt.barh(y + 0.19, va, color=GREY, height=0.36, label="Development persons only (same median width)")
    plt.barh(y - 0.19, vb, color=BLUE, height=0.36, label="Development + PPG-Sport persons")
    if col == "coverage":
        plt.axvline(90, color=MUTED, lw=0.8, ls="--", zorder=0)
    off = 0.02 * xmax
    for yi, v1, v2 in zip(y, va, vb):
        plt.text(v1 + off, yi + 0.19, f"{v1:.0f}", va="center", fontsize=5.5, color=INK)
        plt.text(v2 + off, yi - 0.19, f"{v2:.0f}", va="center", fontsize=5.5, color=INK)
    plt.yticks(y, order, fontsize=7)
    plt.xlim(0, xmax); plt.xticks(ticks)
    plt.xlabel(lab, fontsize=7)
    plt.title(["(a) Coverage by activity", "(b) Width by activity"][k], fontsize=8, color=INK)
plt.figlegend(*plt.gca().get_legend_handles_labels(), frameon=False, fontsize=7, ncol=2, loc="upper center")
plt.tight_layout(rect=(0, 0, 1, 0.88))
plt.savefig(OUT / "crosssensor.pdf"); plt.savefig(OUT / "crosssensor.png", dpi=200)
print(A[["coverage", "median width"]].round(3)); print(B[["coverage", "median width"]].round(3))
