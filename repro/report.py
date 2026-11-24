"""Aggregate results vs. paper (Table 1 / classification) and make figures."""
import glob, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from data import ROOT

R = ROOT / "results"
PAPER = pd.DataFrame([
    ("Baseline", "left", 0.00, 0.76, 0.96), ("Baseline", "right", 0.00, 0.73, 0.92),
    ("Manual-RF", "left", 0.79, 0.32, 0.42), ("Manual-RF", "right", 0.74, 0.32, 0.45),
    ("Roost", "left", 0.89, 0.19, 0.31), ("Roost", "right", 0.85, 0.21, 0.34),
], columns=["model", "target", "R2", "MAE", "RMSE"]).set_index(["model", "target"])

rf = pd.read_csv(R / "rf_metrics.csv")
ro = pd.DataFrame([json.load(open(f)) for f in glob.glob(str(R / "metrics_roost_*.json"))])
reg = ro[ro.task != "cls"].rename(columns={"task": "target"})
allreg = pd.concat([rf, reg[["model", "target", "seed", "R2", "MAE", "RMSE"]]])
g = allreg.groupby(["model", "target"])[["R2", "MAE", "RMSE"]]
mean, std = g.mean(), g.std()
lines = ["## Regression (test set, mean ± std over %d seeds; paper in brackets)\n" % allreg.seed.nunique(),
         "| Model | Endpoint | R² | MAE (V) | RMSE (V) |", "|---|---|---|---|---|"]
for idx in PAPER.index:
    if idx not in mean.index:
        continue
    cells = [f"{mean.loc[idx, m]:.3f} ± {std.loc[idx, m]:.3f} [{PAPER.loc[idx, m]:.2f}]"
             for m in ["R2", "MAE", "RMSE"]]
    lines.append(f"| {idx[0]} | {idx[1]} | " + " | ".join(cells) + " |")

cls = ro[ro.task == "cls"]
if len(cls):
    cols = ["n_test", "n_pos_test", "accuracy", "majority_acc", "precision_w", "recall_w",
            "F1_w", "AUC", "precision_1", "recall_1", "F1_1"]
    cm, cs = cls.groupby("data")[cols].mean(), cls.groupby("data")[cols].std()
    lines += ["\n## Classification ECW ≥ 3 V (paper: acc/prec/rec/F1 ≈ 0.99, AUC 0.95)\n",
              "| Data | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for d in cm.index:
        lines.append(f"| {d} | " + " | ".join(
            f"{cm.loc[d, c]:.3f} ± {cs.loc[d, c]:.3f}" if c not in ("n_test", "n_pos_test")
            else f"{cm.loc[d, c]:.0f}" for c in cols) + " |")
(R / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# parity plots, seed 0: RF and Roost
fig, axs = plt.subplots(2, 2, figsize=(9, 9))
rfp = pd.read_csv(R / "rf_preds_s0.csv")
for j, tgt in enumerate(["left", "right"]):
    for i, (name, df) in enumerate([("Manual-RF", rfp[rfp.target == tgt]),
                                    ("Roost", pd.read_csv(R / f"preds_roost_{tgt}_dedup_s0.csv")
                                     if (R / f"preds_roost_{tgt}_dedup_s0.csv").exists() else None)]):
        ax = axs[i, j]
        if df is None:
            ax.axis("off"); continue
        y, p = df.y.values, df.pred.values
        ax.scatter(y, p, s=6, alpha=0.4)
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        mae = np.abs(y - p).mean(); r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ax.set_title(f"{name} – {tgt} endpoint\nR²={r2:.3f}  MAE={mae:.3f} V")
        ax.set_xlabel("GPPD ECW limit (V vs Li/Li+)"); ax.set_ylabel("Predicted (V)")
fig.tight_layout(); fig.savefig(R / "parity_s0.png", dpi=150)

# ROC curves for classifiers, seed 0
fig, ax = plt.subplots(figsize=(5, 5))
for d in ["dedup", "all"]:
    f = R / f"preds_roost_cls_{d}_s0.csv"
    if f.exists():
        df = pd.read_csv(f); fpr, tpr, _ = roc_curve(df.y, df.prob)
        auc = cls[(cls.data == d) & (cls.seed == 0)].AUC.values[0]
        ax.plot(fpr, tpr, label=f"{d} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("Roost ECW≥3 V classifier, seed 0"); ax.legend()
fig.tight_layout(); fig.savefig(R / "roc_s0.png", dpi=150)

# RF feature importances
imp = pd.read_csv(R / "rf_importances.csv", index_col=0)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
for ax, tgt in zip(axs, ["left", "right"]):
    s = imp[[c for c in imp.columns if c.startswith(tgt)]].mean(1).nlargest(15)[::-1]
    ax.barh([n.replace("MagpieData ", "") for n in s.index], s.values)
    ax.set_title(f"Manual-RF importances – {tgt}")
fig.tight_layout(); fig.savefig(R / "rf_importance.png", dpi=150)
