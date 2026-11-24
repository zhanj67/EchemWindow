"""Baseline (mean predictor) + Manual-RF (Magpie + curated composition features)."""
import json, warnings
import numpy as np, pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import (ElementProperty, Stoichiometry,
                                              ValenceOrbital, IonProperty)
from matminer.featurizers.base import MultipleFeaturizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from data import load, split, ROOT, LEFT, RIGHT, SEEDS
warnings.filterwarnings("ignore")

FEAT_CACHE = ROOT / "splits/magpie_features.csv"


def featurize():
    if FEAT_CACHE.exists():
        return pd.read_csv(FEAT_CACHE, index_col=0)
    df = load(dedup=True)
    comps = pd.DataFrame({"comp": [Composition(f) for f in df.composition]},
                         index=df.composition)
    mf = MultipleFeaturizer([ElementProperty.from_preset("magpie"), Stoichiometry(),
                             ValenceOrbital(props=["frac"]),
                             IonProperty(fast=True)])
    X = mf.featurize_dataframe(comps, "comp", ignore_errors=True, pbar=False)
    X = X.drop(columns="comp").astype(float).fillna(0.0)
    X.to_csv(FEAT_CACHE)
    return X


def metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=mean_squared_error(y, p) ** 0.5)


def main():
    X = featurize()
    print("features:", X.shape)
    rows, imps, preds = [], [], []
    for seed in SEEDS:
        tr, te = split(seed)
        Xtr, Xte = X.loc[tr.composition].values, X.loc[te.composition].values
        for tgt, name in [(LEFT, "left"), (RIGHT, "right")]:
            ytr, yte = tr[tgt].values, te[tgt].values
            rows.append(dict(model="Baseline", target=name, seed=seed,
                             **metrics(yte, np.full_like(yte, ytr.mean()))))
            rf = RandomForestRegressor(n_estimators=500, n_jobs=16, random_state=seed)
            rf.fit(Xtr, ytr); p = rf.predict(Xte)
            rows.append(dict(model="Manual-RF", target=name, seed=seed, **metrics(yte, p)))
            imps.append(pd.Series(rf.feature_importances_, index=X.columns, name=f"{name}_s{seed}"))
            if seed == 0:
                preds.append(pd.DataFrame(dict(composition=te.composition, target=name,
                                               y=yte, pred=p)))
            print(rows[-1])
    pd.DataFrame(rows).to_csv(ROOT / "results/rf_metrics.csv", index=False)
    pd.concat(imps, axis=1).to_csv(ROOT / "results/rf_importances.csv")
    pd.concat(preds).to_csv(ROOT / "results/rf_preds_s0.csv", index=False)


if __name__ == "__main__":
    main()
