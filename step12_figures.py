"""Figures of the paper."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures"); OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, INK, MUTED, GRID = "#2a78d6", "#3a9e4f", "#1f1f1d", "#6b6a63", "#e1e0d9"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})

DALIA = pd.read_csv("results/intervals_DaLiA_normalized.csv")
WESAD = pd.read_csv("results/intervals_WESAD_normalized.csv")
GALAXY = pd.read_csv("results/intervals_galaxy_normalized.csv")
BELIEF = pd.read_csv("results/intervals_galaxy_beliefppg.csv")


def covered(r, w):
    return (r.hr_ref >= r[f"lo_{w}"]) & (r.hr_ref <= r[f"hi_{w}"])


# Figure: coverage of each person with the three calibrations (normalized score)
NAVY = "#104281"   # person-quantile; palette green/blue/navy validated on all pairs (validate_palette.js)
PQ = pd.read_csv("results/person_coverage_by_calibration.csv")
cases = [("PPG-DaLiA", DALIA, "PPG-DaLiA"), ("WESAD", WESAD, "WESAD"),
         ("GalaxyPPG, MAST", GALAXY, "GalaxyPPG"), ("GalaxyPPG, BeliefPPG", BELIEF, "BeliefPPG")]
def swarm(v, width=0.0065, step=0.12):
    """Vertical offsets so that markers closer than `width` in coverage do not overlap."""
    order = np.argsort(v); off = np.zeros(len(v)); placed = []
    for i in order:
        for d in [0, step, -step, 2 * step, -2 * step, 3 * step, -3 * step]:
            if all(abs(v[i] - v[j]) > width or abs(d - off[j]) > step * 0.9 for j in placed):
                off[i] = d; break
        placed.append(i)
    return off


plt.figure(figsize=(7.16, 2.5))
rows = [("window", "Window-level", GREEN, "o"), ("subject", "Person-level", BLUE, "o"), ("quantile", "Person-quantile", NAVY, "D")]
for k, (name, r, key) in enumerate(cases):
    cov = {w: covered(r, w).groupby(r.subject).mean() for w in ("window", "subject")}
    cov["quantile"] = PQ[(PQ.setting == key) & (PQ.calibration == "person-quantile")].set_index("subject").coverage
    subjects = cov["window"].index
    n = len(subjects)
    ax = plt.subplot(1, 4, k + 1)
    plt.axvline(0.9, color=MUTED, lw=0.8, ls="--", zorder=0)
    pos = {}
    for i, (w, _, _, _) in enumerate(rows):
        v = cov[w][subjects].to_numpy()
        pos[w] = 2 - i + swarm(v)
    for j in range(n):  # one light line per participant across the three calibrations
        plt.plot([cov[w][subjects[j]] for w, _, _, _ in rows], [pos[w][j] for w, _, _, _ in rows], color=GRID, lw=0.5, zorder=1)
    for i, (w, lab, col, mk) in enumerate(rows):
        v = cov[w][subjects].to_numpy(); below = v < 0.9
        plt.scatter(v[~below], pos[w][~below], s=12, marker=mk, facecolors="white", edgecolors=col, linewidths=0.8, zorder=3, clip_on=False)
        plt.scatter(v[below], pos[w][below], s=14, marker=mk, color=col, zorder=4, clip_on=False)
        plt.text(1.08, 2 - i, f"{(~below).sum()}/{n}", fontsize=6, color=INK, va="center", ha="left",
                 transform=ax.get_yaxis_transform(), clip_on=False)
    plt.yticks([2, 1, 0], [lab for _, lab, _, _ in rows] if k == 0 else ["", "", ""], fontsize=6.5)
    plt.ylim(-0.55, 2.55); plt.xlim(0.72, 1.0)
    plt.xticks([0.8, 0.9, 1.0], ["80%", "90%", "100%"], fontsize=6.5)
    plt.title(name, fontsize=7.5, color=INK)
    plt.xlabel("Coverage of each participant", fontsize=6.5)
plt.tight_layout(pad=0.4, w_pad=3.6)
plt.savefig(OUT / "coverage_per_person.pdf"); plt.savefig(OUT / "coverage_per_person.png", dpi=200)
plt.close()

# Figure: coverage per condition, person-level vs motion-stratified calibration (normalized score)
MAGENTA = "#c9338a"   # motion-stratified; green/blue/navy/magenta checked on all pairs for CVD (validate_palette.js)
galmap = {"baseline": "rest", "rest": "rest", "meditation": "rest", "adaptation": "rest", "screen reading": "rest",
          "standing": "rest", "TSST preparation": "stress", "TSST speech": "stress", "SSST preparation": "stress",
          "SSST singing": "stress", "keyboard typing": "typing", "mobile typing": "typing",
          "walking": "walking", "jogging": "jogging", "running": "running"}
ACT = pd.read_csv("results/rules_activity.csv")
panels = [("PPG-DaLiA", ["sitting", "working", "driving", "lunch", "walking", "cycling", "table soccer", "stairs"]),
          ("WESAD", ["baseline", "meditation", "amusement", "stress"]),
          ("GalaxyPPG", ["rest", "typing", "stress", "walking", "jogging", "running"])]
plt.figure(figsize=(7.16, 2.5))
for k, (name, order) in enumerate(panels):
    a = ACT[ACT.setting == name]
    pl = a[a.rule == "person-level"].set_index("condition").coverage.reindex(order) * 100
    ms = a[a.rule == "motion-stratified"].set_index("condition").coverage.reindex(order) * 100
    y = np.arange(len(order))[::-1]
    plt.subplot(1, 3, k + 1)
    plt.barh(y + 0.19, pl, color=BLUE, height=0.36, label="person-level")
    plt.barh(y - 0.19, ms, color=MAGENTA, height=0.36, label="motion-stratified")
    plt.axvline(90, color=MUTED, lw=0.8, ls="--", zorder=0)
    for yi, v1, v2 in zip(y, pl, ms):
        plt.text(104, yi + 0.19, f"{v1:.0f}", va="center", fontsize=5.5, color=INK)
        plt.text(104, yi - 0.19, f"{v2:.0f}", va="center", fontsize=5.5, color=INK)
    plt.yticks(y, order, fontsize=7)
    plt.xlim(0, 116)
    plt.xticks([0, 50, 90], ["0", "50", "90"])
    plt.title(name, fontsize=8, color=INK)
    plt.xlabel("Coverage (%)", fontsize=7)
plt.figlegend(*plt.gca().get_legend_handles_labels(), frameon=False, fontsize=7, ncol=2, loc="upper center")
plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig(OUT / "coverage_by_condition.pdf"); plt.savefig(OUT / "coverage_by_condition.png", dpi=200)
plt.close()

# Figure: coverage-width trade-off. Line: one global threshold multiplying sigma for all participants.
CURVE = pd.read_csv("results/rules_curve.csv"); RULES_T = pd.read_csv("results/rules_summary.csv")
marks = {"window-level": (GREEN, "o", "window"), "person-level": (BLUE, "o", "person"),
         "person-quantile": (NAVY, "D", "quantile"), "motion-stratified": (MAGENTA, "s", "motion")}
# label offsets (bpm, persons) chosen so that labels do not overlap
offsets = {"Both": {"window-level": (-0.8, 1.2, "right"), "person-level": (-0.8, 1.0, "right"),
                    "person-quantile": (0.8, -2.3, "left"), "motion-stratified": (0.8, -2.6, "left")},
           "GalaxyPPG": {"window-level": (0.8, -1.4, "left"), "person-level": (0.8, -1.6, "left"),
                         "person-quantile": (0.8, -1.6, "left"), "motion-stratified": (-0.7, 0.6, "right")}}
plt.figure(figsize=(3.5, 2.25))
for k, (setting, title, xlim) in enumerate([("Both", "Development (LOSO)", (14, 42)), ("GalaxyPPG", "GalaxyPPG", (5, 30))]):
    plt.subplot(1, 2, k + 1)
    c = CURVE[CURVE.setting == setting]
    n = int(c.persons.iloc[0])
    plt.axhline(n, color=GRID, lw=0.8, ls="--", zorder=0)
    plt.plot(c["median width"], c["at target"], color=MUTED, lw=1.1, zorder=1)
    for rule, (col, mk, lab) in marks.items():
        r = RULES_T[(RULES_T.setting == setting) & (RULES_T.rule == rule)].iloc[0]
        x, y = r["median width"], r["at target"]
        plt.scatter(x, y, color=col, marker=mk, s=26, zorder=3, edgecolors="white", linewidths=0.6)
        dx, dy, ha = offsets[setting][rule]
        plt.text(x + dx, y + dy, lab, fontsize=6, color=INK, ha=ha, va="center")
    xt = xlim[0] + 0.55 * (xlim[1] - xlim[0])
    plt.text(xlim[1], n + 0.6, f"all {n}", fontsize=5.5, color=MUTED, ha="right", va="bottom")
    plt.xlim(*xlim); plt.ylim(0, n + 3)
    plt.title(title, fontsize=7, color=INK)
    plt.xlabel("Median width (bpm)", fontsize=6.5)
    if k == 0:
        plt.ylabel("Participants at target", fontsize=6.5)
    plt.xticks(fontsize=6); plt.yticks(fontsize=6)
plt.subplot(1, 2, 2)
plt.annotate("one global\nthreshold", xy=(11.5, 6.5), xytext=(20, 6), fontsize=5.5, color=MUTED, ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
plt.tight_layout(pad=0.4, w_pad=1.0)
plt.savefig(OUT / "coverage_width.pdf"); plt.savefig(OUT / "coverage_width.png", dpi=250)
plt.close()

# Figure: example recording, GalaxyPPG participant 12, the participant with the smallest summed
# standardized distance to the medians of coverage, median width and MAE over participants
phase = {"baseline": "rest", "rest": "rest", "meditation": "meditation", "adaptation": "rest",
         "screen reading": "reading", "standing": "standing", "TSST preparation": "TSST", "TSST speech": "TSST",
         "SSST preparation": "SSST", "SSST singing": "SSST", "keyboard typing": "typing", "mobile typing": "typing",
         "walking": "walking", "jogging": "jogging", "running": "running"}
s = GALAXY[GALAXY.subject == "G12"].reset_index(drop=True)
t = s.window * 2 / 60
plt.figure(figsize=(7.16, 2.2))
plt.fill_between(t, s.lo_subject, s.hi_subject, color=BLUE, alpha=0.25, lw=0, label="90% interval")
plt.plot(t, s.hr_hat, color=GREEN, lw=0.8, label="estimated HR")
plt.plot(t, s.hr_ref, color=INK, lw=0.8, label="ECG reference")
labels = s.condition.map(phase).fillna("transition").to_numpy()
starts = [a for a in np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]]) if labels[a] != "transition"]
shown = []
for a in starts:
    if labels[a] == "rest" and "rest" in shown:
        continue
    plt.text(t[a], 198 - 12 * (len(shown) % 2), labels[a], fontsize=6, color=MUTED, va="top")
    shown.append(labels[a])
plt.ylim(40, 200)
plt.xlabel("Time (min)")
plt.ylabel("HR (bpm)")
plt.legend(frameon=False, fontsize=6, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.16))
plt.tight_layout()
plt.savefig(OUT / "example_G12.pdf"); plt.savefig(OUT / "example_G12.png", dpi=200)
plt.close()
print("ok")
