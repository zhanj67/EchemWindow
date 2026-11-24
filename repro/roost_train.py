"""Roost (aviary) regression of ECW left/right endpoints and >=3 V classification.

usage: python roost_train.py --task {left,right,cls} --seed 0 [--data dedup|all]
"""
import argparse, json
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, precision_recall_fscore_support, roc_auc_score)
import aviary, aviary.core, aviary.utils
from aviary.roost.data import CompositionData, collate_batch
from aviary.roost.model import Roost
from aviary.utils import train_ensemble, results_multitask
from data import split, ROOT, LEFT, RIGHT

# keep checkpoints inside the project, not site-packages
aviary.ROOT = aviary.core.ROOT = aviary.utils.ROOT = str(ROOT)

# aviary bug: non-robust CSE gets (B,1) targets -> flatten them for CrossEntropyLoss
_orig_init_losses = aviary.utils.initialize_losses
def _init_losses(task_dict, loss_name_dict, robust=False):
    d = _orig_init_losses(task_dict, loss_name_dict, robust)
    for k, (t, f) in d.items():
        if t == "classification" and not robust:
            ce = torch.nn.CrossEntropyLoss()
            d[k] = (t, lambda o, y, ce=ce: ce(o, y.view(-1).long()))
    return d
aviary.utils.initialize_losses = _init_losses

ap = argparse.ArgumentParser()
ap.add_argument("--task", choices=["left", "right", "cls"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", choices=["dedup", "all"], default="dedup")
ap.add_argument("--epochs", type=int, default=None)  # 300 reg / 150 cls
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--wd", type=float, default=None)  # 1e-6 reg / 1e-4 cls
ap.add_argument("--batch", type=int, default=128)
args = ap.parse_args()
_cls = args.task == "cls"
if args.epochs is None: args.epochs = 150 if _cls else 300
if args.wd is None: args.wd = 1e-4 if _cls else 1e-6

torch.manual_seed(args.seed); np.random.seed(args.seed)
dedup = args.data == "dedup"
is_cls = args.task == "cls"
tr, te = split(args.seed, dedup=dedup, stratify=is_cls)
target = {"left": LEFT, "right": RIGHT, "cls": "label"}[args.task]
task_dict = {target: "classification" if is_cls else "regression"}
loss_dict = {target: "CSE" if is_cls else "L1"}

# 10% of training set held out for early-stopping checkpoint selection
tr, va = train_test_split(tr, test_size=0.1, random_state=args.seed,
                          stratify=tr["label"] if is_cls else None)
mk = lambda d: CompositionData(d.reset_index(drop=True), task_dict,
                               identifiers=("material_id", "composition"))
loader = lambda d, sh: DataLoader(mk(d), batch_size=args.batch, shuffle=sh,
                                  collate_fn=collate_batch)
train_loader, val_loader = loader(tr, True), loader(va, False)
test_loader = DataLoader(mk(te), batch_size=1024, shuffle=False, collate_fn=collate_batch)

model_name = f"roost_{args.task}_{args.data}_s{args.seed}"
device = "cuda" if torch.cuda.is_available() else "cpu"
ROBUST = not is_cls  # robust (sampled-softmax) cls head gave NaN logits
model_params = dict(task_dict=task_dict, robust=ROBUST,
                    n_targets=[2] if is_cls else [1],
                    elem_embedding="matscholar200", elem_fea_len=64, n_graph=3,
                    elem_heads=3, elem_gate=[256], elem_msg=[256],
                    cry_heads=3, cry_gate=[256], cry_msg=[256],
                    trunk_hidden=[1024, 512], out_hidden=[256, 128, 64])
setup_params = dict(optim="AdamW", learning_rate=args.lr, weight_decay=args.wd,
                    momentum=0.9, device=device)
restart_params = dict(resume=None, fine_tune=None, transfer=None)

if not is_cls:
    train_ensemble(model_class=Roost, model_name=model_name, run_id=0, ensemble_folds=1,
                   epochs=args.epochs, train_loader=train_loader, val_loader=val_loader,
                   log=None, setup_params=setup_params, restart_params=restart_params,
                   model_params=model_params, loss_dict=loss_dict, verbose=False)
else:
    # Classifier: 300 epochs @ wd 1e-6 overfit to train loss ~0 then diverged to NaN.
    # Train epoch by epoch, keep the weights with the best validation ROC-AUC
    # (paper also used AUC as its tuning metric) and stop if the loss goes NaN.
    import copy
    m = aviary.utils.initialize_model(model_class=Roost, model_params=model_params,
                                      **setup_params, **restart_params)
    optimizer, scheduler = aviary.utils.initialize_optim(m, **setup_params, **restart_params)
    loss_fns = aviary.utils.initialize_losses(m.task_dict, loss_dict, ROBUST)
    norms = {target: None}
    best_auc, best_ep, best_state = -1.0, -1, None
    for ep in range(args.epochs):
        m.fit(train_loader, None, optimizer, scheduler, epochs=1, loss_dict=loss_fns,
              normalizer_dict=norms, model_name=model_name, run_id=0,
              checkpoint=False, writer=None, verbose=False)
        yv, ov, *_ = m.predict(val_loader)
        pv = torch.softmax(torch.as_tensor(ov[0]).float().cpu(), 1)[:, 1].numpy()
        if not np.isfinite(pv).all():
            print(f"non-finite outputs at epoch {ep}; stopping"); break
        auc = roc_auc_score(np.asarray(yv[0]).ravel().astype(int), pv)
        if auc > best_auc:
            best_auc, best_ep, best_state = auc, ep, copy.deepcopy(m.state_dict())
    print(f"best val AUC {best_auc:.4f} at epoch {best_ep}")
    m.load_state_dict(best_state)
    (ROOT / f"models/{model_name}").mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_params=model_params, state_dict=best_state, best_epoch=best_ep,
                    best_val_AUC=best_auc), ROOT / f"models/{model_name}/best_auc.pt")

if not is_cls:
    res = results_multitask(model_class=Roost, model_name=model_name, run_id=0,
                            ensemble_folds=1, test_loader=test_loader, robust=ROBUST,
                            task_dict=task_dict, device=device, eval_type="checkpoint",
                            print_results=True, save_results=False)[target]

out = dict(model="Roost", task=args.task, data=args.data, seed=args.seed,
           n_train=len(tr), n_val=len(va), n_test=len(te), epochs=args.epochs, wd=args.wd)
if is_cls:
    out.update(best_val_AUC=best_auc, best_epoch=best_ep)
if is_cls:
    # aviary's non-robust cls eval calls ndarray.softmax (bug) -> evaluate checkpoint here.
    yt, outs, *_ = m.predict(test_loader)
    y = np.asarray(yt[0]).ravel().astype(int)
    prob = torch.softmax(torch.as_tensor(outs[0]).float().cpu(), 1)[:, 1].numpy()
    pred = (prob >= 0.5).astype(int)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    p1, r1, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    out.update(n_pos_test=int(y.sum()), accuracy=accuracy_score(y, pred), AUC=roc_auc_score(y, prob),
               precision_w=pw, recall_w=rw, F1_w=fw, precision_1=p1, recall_1=r1, F1_1=f1,
               majority_acc=1 - y.mean())
    pd.DataFrame(dict(composition=te.composition, y=y, prob=prob)).to_csv(
        ROOT / f"results/preds_{model_name}.csv", index=False)
else:
    y = np.asarray(res["targets"]).ravel(); p = res["preds"][0]
    out.update(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
               RMSE=mean_squared_error(y, p) ** 0.5)
    pd.DataFrame(dict(composition=te.composition, y=y, pred=p)).to_csv(
        ROOT / f"results/preds_{model_name}.csv", index=False)
print("RESULT", json.dumps(out))
with open(ROOT / f"results/metrics_{model_name}.json", "w") as fh:
    json.dump(out, fh, indent=1)
