# Reproduction of Chen et al., J. Mater. Chem. A 2025, 13, 23445 (ECW ML)

Data: `../ecw_mp_16k.csv` (Zenodo 15505298) + SI Table S6 (318 OQMD compounds, parsed to
`splits/oqmd318_tableS6.csv`). Env: `conda activate ecw` (torch 2.4.1+cu118, aviary/Roost, matminer).

## Pipeline
| Script | What |
|---|---|
| `data.py` | dedup by formula (5 939 unique), 80/20 split, seeds 0–4 (cached in `splits/`) |
| `rf_magpie.py` | Baseline (train mean) + Manual-RF (Magpie+Stoichiometry+ValenceOrbital+IonProperty, 500 trees) |
| `roost_train.py` | Roost: `--task left|right|cls --seed N [--data dedup|all]` |
| `run_all.sh` | runs all missing Roost jobs over 3 GPUs (resumable) |
| `report.py` | `results/summary.md`, parity / ROC / feature-importance plots |
| `predict_oqmd.py` | external test on the 318 OQMD compounds vs. paper's Table S6 predictions |

Roost settings (no Optuna; paper's hyper-parameters are not reported): matscholar200, default
architecture, AdamW lr 3e-4, batch 128, 10 % of train as validation.
Regression: robust L1, 300 epochs, wd 1e-6, final-epoch model.
Classification: plain cross-entropy, 150 epochs, wd 1e-4, weights with best validation AUC kept,
stop on non-finite outputs (300 ep / wd 1e-6 overfit and diverged to NaN; old runs archived in
`results/cls_v1_overfit/`, `results/cls_v2_nosave/`).
aviary bugs worked around in-script (no site-packages edits): non-robust CSE target shape,
`ndarray.softmax` in eval, "best" checkpoint keeping the *lowest* accuracy.

## Results (test set, mean ± std over 5 random 80/20 splits; paper in brackets)
| Model | Endpoint | R² | MAE (V) | RMSE (V) |
|---|---|---|---|---|
| Baseline | left / right | 0.00 / 0.00 [0.00/0.00] | 0.76 / 0.73 [0.76/0.73] | 0.97 / 0.93 [0.96/0.92] |
| Manual-RF | left | 0.836 ± 0.014 [0.79] | 0.275 [0.32] | 0.393 [0.42] |
| Manual-RF | right | 0.821 ± 0.005 [0.74] | 0.269 [0.32] | 0.393 [0.45] |
| Roost | left | 0.872 ± 0.017 [0.89] | 0.190 ± 0.011 [0.19] | 0.347 ± 0.026 [0.31] |
| Roost | right | 0.833 ± 0.024 [0.85] | 0.203 ± 0.017 [0.21] | 0.378 ± 0.034 [0.34] |

Classification ECW ≥ 3 V, dedup data (14 positives per test set):
acc 0.989, weighted P/R/F1 0.983/0.989/0.985 — **paper Table S5: 0.989/0.989/0.989/0.985** (match).
ROC-AUC 0.897 ± 0.040 [0.95]; recall of Class 1 only 0.17 ± 0.16 (always-Class-0 accuracy = 0.988).
Trained on all 16k rows instead: AUC 0.986, acc 0.995 — but ~73 % of test rows share a formula with
the training set, and acc 0.995 contradicts Table S5, so the paper evidently used the dedup data.

External OQMD set (318 compounds, 0 formula overlap with MP data):
| | left R² / MAE | right R² / MAE |
|---|---|---|
| Ours, 5-seed ensemble | 0.901 / 0.186 | 0.854 / 0.239 |
| Ours, single seeds | 0.79–0.90 | 0.75–0.86 |
| Paper (Table S6 preds) | 0.873 / 0.198 | 0.817 / 0.321 (text: R² 0.90 / 0.82) |

| OQMD classifier | TP/21 | FP | F1(Class 1) | AUC |
|---|---|---|---|---|
| Paper Roost-IMB | 13 | 3 | 0.70 | 0.80 (hard labels) |
| Ours Roost-cls, seeds 0/1/3 | 14/10/13 | 4/6/2 | 0.72/0.54/0.72 | 0.94–0.97 (prob) |
| Ours Roost-cls, seeds 2/4 | 0 | 0 | 0 | 0.98/0.94 (prob) — never exceeds p=0.5 |
| Ours regression width ≥ 3 V | **19** | 11 | **0.75** | 0.99 (continuous width) |

## Conclusions
1. Regression is reproduced: Roost MAE matches Table 1; R² within ~1 σ of split-to-split scatter.
   Manual-RF is slightly better than reported. Covalent radius / electronegativity dominate RF
   importances, as in Fig. S6.
2. Classifier reproduces Table S5 exactly, but those metrics ≈ the trivial majority baseline.
   AUC 0.95 is at the upper end of our seed range (0.85–0.95); with 14 positives per test set the
   AUC itself has ±0.04 scatter. The 0.5 threshold is badly calibrated: 2/5 seeds predict no positives.
3. OQMD transfer is reproduced (ensemble ≥ paper). The paper's OQMD AUCs (0.80 / 0.61) were computed
   from hard 0/1 labels (we reproduce both from Table S6).
4. Practical takeaway: thresholding the regression-predicted width at 3 V is a better wide-ECW screen
   than the dedicated classifier (recall 19/21 vs 13/21 on OQMD).

Not reproduced: Auto-RF and CrabNet (out of scope); paper's exact split seed and tuned hyper-parameters
are unpublished.
