"""External test: our Roost models on the 318 DFT-labelled OQMD compounds (SI Table S6).

Compares against the paper's own predictions listed in Table S6 (Fig. S12 / S15).
"""
import json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             roc_auc_score, accuracy_score, precision_recall_fscore_support)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from aviary.data import Normalizer
from aviary.roost.data import CompositionData, collate_batch
from aviary.roost.model import Roost
from data import ROOT, LEFT, RIGHT, SEEDS

dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(ROOT / "splits/oqmd318_tableS6.csv")
df["material_id"] = df.oqmd_id.astype(str)
df[LEFT], df[RIGHT] = df.dft_left, df.dft_right


def ctype(f):
    els = {e.symbol for e in Composition(f).elements}
    hal = els & {"F", "Cl", "Br", "I"}
    if "O" in els and not hal and "S" not in els: return "oxide"
    if hal and not els & {"O", "S", "N"}: return "halide"
    return "other"


df["ctype"] = df.composition.map(ctype)


def run(ck, target, task):
    model = Roost(**ck["model_params"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    ds = CompositionData(df, {target: task}, identifiers=("material_id", "composition"))
    _, out, *_ = model.predict(DataLoader(ds, batch_size=1024, collate_fn=collate_batch))
    o = torch.as_tensor(out[0]).float().cpu()
    if task == "classification":
        return torch.softmax(o, 1)[:, 1].numpy()
    norm = Normalizer.from_state_dict(ck["normalizer_dict"][target])
    mean = o[:, 0] if o.ndim == 2 else o
    return norm.denorm(mean).numpy()


def reg_metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=mean_squared_error(y, p) ** 0.5)


def cls_metrics(y, prob=None, pred=None):
    pred = (prob >= 0.5).astype(int) if pred is None else pred
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    p1, r1, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return dict(AUC_prob=roc_auc_score(y, prob) if prob is not None else np.nan,
                AUC_hard=roc_auc_score(y, pred), accuracy=accuracy_score(y, pred),
                F1_w=fw, precision_1=p1, recall_1=r1, F1_1=f1,
                TP=int(((pred == 1) & (y == 1)).sum()), FP=int(((pred == 1) & (y == 0)).sum()))


rows = []
for s in SEEDS:
    for tgt, name in [(LEFT, "left"), (RIGHT, "right")]:
        ck = torch.load(ROOT / f"models/roost_{name}_dedup_s{s}/checkpoint-r0.pth.tar",
                        map_location=dev, weights_only=False)
        df[f"ours_{name}_s{s}"] = p = run(ck, tgt, "regression")
        rows.append(dict(model="ours", target=name, seed=s, **reg_metrics(df[tgt], p)))
    f = ROOT / f"models/roost_cls_dedup_s{s}/best_auc.pt"
    if f.exists():
        ck = torch.load(f, map_location=dev, weights_only=False)
        df[f"ours_prob_s{s}"] = prob = run(ck, "label", "classification")
        rows.append(dict(model="ours", target="cls", seed=s, **cls_metrics(df.label.values, prob)))
for name in ["left", "right"]:
    df[f"ours_{name}_ens"] = df[[f"ours_{name}_s{s}" for s in SEEDS]].mean(1)
    rows.append(dict(model="ours-ens5", target=name, **reg_metrics(df["dft_" + name], df[f"ours_{name}_ens"])))
    rows.append(dict(model="paper", target=name, **reg_metrics(df["dft_" + name], df["paper_" + name])))
    for ct in ["oxide", "halide"]:
        d = df[df.ctype == ct]
        rows.append(dict(model=f"ours-ens5/{ct}(n={len(d)})", target=name,
                         **reg_metrics(d["dft_" + name], d[f"ours_{name}_ens"])))
        rows.append(dict(model=f"paper/{ct}(n={len(d)})", target=name,
                         **reg_metrics(d["dft_" + name], d["paper_" + name])))
rows.append(dict(model="paper Roost-IMB", target="cls", **cls_metrics(df.label.values, pred=df.paper_imb.values)))
rows.append(dict(model="paper Roost-BAL", target="cls", **cls_metrics(df.label.values, pred=df.paper_bal.values)))
# also: does thresholding the regression width at 3 V work as a classifier?
w = df.ours_right_ens - df.ours_left_ens
rows.append(dict(model="ours-ens5 width>=3V", target="cls",
                 **cls_metrics(df.label.values, prob=w.values, pred=(w >= 3).astype(int).values)))

res = pd.DataFrame(rows)
res.to_csv(ROOT / "results/oqmd_metrics.csv", index=False)
df.to_csv(ROOT / "results/oqmd_preds.csv", index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(res.round(3).to_string(index=False))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for ax, name in zip(axs, ["left", "right"]):
    y = df["dft_" + name]
    ax.scatter(y, df["paper_" + name], s=10, alpha=0.5, label="paper (Table S6)")
    ax.scatter(y, df[f"ours_{name}_ens"], s=10, alpha=0.5, label="ours (5-seed mean)")
    lo, hi = y.min() - 0.2, y.max() + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(f"OQMD external set – {name} endpoint (n={len(df)})")
    ax.set_xlabel("DFT ECW limit (V)"); ax.set_ylabel("Predicted (V)"); ax.legend()
fig.tight_layout(); fig.savefig(ROOT / "results/oqmd_parity.png", dpi=150)
