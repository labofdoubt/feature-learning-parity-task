from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/root/parity/analysis_output/accessibility_4bit")
df = pd.read_csv(OUT / "results.csv")
RC = 1/np.sqrt(8)
print(df[["seed","alpha","R","Q4_frozen","loss","acc","n_rel","frac22_w","frac1111_w","n_22","n_1111"]]
      .round(3).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, lab in ((axes[0], "frac22_w", "fraction of 2+2→4 computation"),
                     (axes[1], "frac1111_w", "fraction of 1+1+1+1→4 computation")):
    for seed, g in df.groupby("seed"):
        g = g.sort_values("R")
        ax.plot(g.R, g[col], "o", alpha=0.45, label=f"seed {seed}" if col=="frac22_w" else None)
    m = df.groupby("alpha").agg(R=("R","mean"), f=(col,"mean")).sort_values("R")
    ax.plot(m.R, m.f, "k-", lw=2.2, label="mean over seeds" if col=="frac22_w" else None)
    ax.axvline(RC, color="r", ls="--", label=r"$R_c=1/\sqrt{8}\approx0.354$" if col=="frac22_w" else None)
    ax.set_xlabel("measured  $R=Q_2/Q_1$"); ax.set_ylabel(lab)
    ax.set_xlim(-0.03, 1.0)
axes[0].legend(fontsize=8)
fig.suptitle("Accessibility transition: which circuit does block 2 learn?  "
             "(4 seeds × 13 α, exact 16-point cube)", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "main_transition.pdf", bbox_inches="tight")
fig.savefig(OUT / "main_transition.png", dpi=170, bbox_inches="tight")
print("replotted")
