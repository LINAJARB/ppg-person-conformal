"""Coverage-width trade-off table: for each calibration, the number of participants at target,
its median width, and the median width that one global threshold needs to bring at least as many
participants to target (from results/rules_curve.csv and results/rules_summary.csv)."""
import pandas as pd

C = pd.read_csv("results/rules_curve.csv"); R = pd.read_csv("results/rules_summary.csv")
rows = []
for st in ["Both", "GalaxyPPG"]:
    c = C[C.setting == st]
    for rule in ["window-level", "person-level", "person-quantile", "motion-stratified"]:
        r = R[(R.setting == st) & (R.rule == rule)].iloc[0]
        need = c[c["at target"] >= r["at target"]]["median width"].min()
        rows.append({"setting": st, "rule": rule, "at target": int(r["at target"]), "persons": int(c.persons.iloc[0]),
                     "median width": r["median width"], "global threshold width": need})
T = pd.DataFrame(rows); T.to_csv("results/tradeoff_table.csv", index=False)
print(T.round(1).to_string(index=False))
