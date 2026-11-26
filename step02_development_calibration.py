"""Step 2: conformal intervals on the development data (DaLiA, WESAD, both)."""
from pathlib import Path
import pandas as pd
from ppgs.conformal import loso, summarize, METHODS

out = Path("results")
df = pd.read_csv(out / "windows.csv")
setups = {"DaLiA": df[df.dataset == "DaLiA"], "WESAD": df[df.dataset == "WESAD"], "DaLiA+WESAD": df}
rows = []
for name, d in setups.items():
    d = d.reset_index(drop=True)
    for m in METHODS:
        res = loso(d, m)
        res.to_csv(out / f"intervals_{name}_{m}.csv", index=False)
        for w in ("window", "subject"):
            groups = [("all", res)] + ([(ds, g) for ds, g in res.groupby("dataset")] if name == "DaLiA+WESAD" else [])
            for gname, g in groups:
                rows.append({"setup": name, "evaluated on": gname, "method": m, "weighting": w, **summarize(g, w)})
        print(name, m, "done", flush=True)
tab = pd.DataFrame(rows)
tab.to_csv(out / "summary.csv", index=False)
pd.set_option("display.width", 250)
print(tab.round(3).to_string(index=False))
