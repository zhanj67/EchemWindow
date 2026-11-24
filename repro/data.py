"""Shared data loading / splitting for the ECW reproduction (Chen et al. 2025)."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
CSV = ROOT.parent / "ecw_mp_16k.csv"
LEFT, RIGHT, WIN = "Left End (V)", "Right End (V)", "Window Length (V)"
SEEDS = [0, 1, 2, 3, 4]


def load(dedup: bool = True) -> pd.DataFrame:
    df = pd.read_csv(CSV, encoding="utf-8-sig")
    df = df.rename(columns={"MP ID": "material_id", "Formula": "composition"})
    if dedup:  # paper: "de-duplicated to retain only unique chemical formulas"
        df = df.drop_duplicates("composition").reset_index(drop=True)
    return df


def split(seed: int, dedup: bool = True, stratify: bool = False):
    """80/20 train/test split; cached to splits/ so all models share test sets."""
    tag = f"{'dedup' if dedup else 'all'}_{'strat' if stratify else 'rand'}_s{seed}"
    f_tr, f_te = ROOT / f"splits/{tag}_train.csv", ROOT / f"splits/{tag}_test.csv"
    if f_tr.exists():
        return pd.read_csv(f_tr), pd.read_csv(f_te)
    df = load(dedup)
    tr, te = train_test_split(df, test_size=0.2, random_state=seed,
                              stratify=df["label"] if stratify else None)
    tr.to_csv(f_tr, index=False); te.to_csv(f_te, index=False)
    return tr.reset_index(drop=True), te.reset_index(drop=True)
