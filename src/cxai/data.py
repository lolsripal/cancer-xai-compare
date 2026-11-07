"""Loading, labelling and splitting TCGA-BRCA gene-expression data.

The UCSC Xena ``HiSeqV2`` matrix is genes x samples with values
log2(RSEM normalized count + 1). Sample type comes from the TCGA barcode:
``TCGA-XX-XXXX-01`` is a primary tumor and ``-11`` is solid tissue normal.
Many normals have a matched tumor from the same patient, so all splits are
grouped by patient to stop the model "recognising the person" instead of the
tissue.
"""

from __future__ import annotations

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

TUMOR_CODE = "01"
NORMAL_CODE = "11"


@dataclass
class Dataset:
    """Train / validation / test splits in a shared, standardized gene space.

    * ``train`` fits the models.
    * ``val`` is where explanations (SHAP, permutation importance) are computed.
    * ``test`` is untouched until evaluation, ablation and sparse-model checks,
      so gene rankings never leak into the numbers we report.
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    genes: list[str]
    scaler: StandardScaler
    raw_train: pd.DataFrame = field(repr=False)  # unscaled log2 expression (for fold changes)
    raw_test: pd.DataFrame = field(repr=False)

    @property
    def X_trainval(self) -> pd.DataFrame:
        return pd.concat([self.X_train, self.X_val])

    @property
    def y_trainval(self) -> pd.Series:
        return pd.concat([self.y_train, self.y_val])

    def summary(self) -> dict:
        def counts(y):
            return {"n": int(len(y)), "tumor": int(y.sum()), "normal": int((1 - y).sum())}

        return {
            "n_genes": len(self.genes),
            "train": counts(self.y_train),
            "val": counts(self.y_val),
            "test": counts(self.y_test),
        }


def download(url: str, path: str | Path) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path}")
    urllib.request.urlretrieve(url, path)
    return path


def load_xena_matrix(path: str | Path) -> pd.DataFrame:
    """Return a samples x genes DataFrame from a Xena genes x samples matrix."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    df = df[~df.index.duplicated(keep="first")]
    return df.T.astype(np.float32)


def label_samples(expr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Keep primary tumors (1) and solid tissue normals (0); return X, y, patient ids."""
    codes = expr.index.str.split("-").str[3].str[:2]
    keep = codes.isin([TUMOR_CODE, NORMAL_CODE])
    X = expr.loc[keep]
    y = pd.Series((codes[keep] == TUMOR_CODE).astype(int), index=X.index, name="tumor")
    patients = pd.Series(X.index.str[:12], index=X.index, name="patient")
    return X, y, patients


def grouped_split(y: pd.Series, groups: pd.Series, n_splits: int, seed: int):
    """One stratified, patient-grouped split: returns (rest_index, holdout_index)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rest, hold = next(sgkf.split(np.zeros(len(y)), y, groups))
    return y.index[rest], y.index[hold]


def prepare(
    X: pd.DataFrame,
    y: pd.Series,
    patients: pd.Series,
    n_genes: int = 2000,
    min_mean_expr: float = 1.0,
    seed: int = 42,
) -> Dataset:
    """Split by patient (64/16/20), select genes on train only, standardize."""
    trainval_idx, test_idx = grouped_split(y, patients, n_splits=5, seed=seed)
    train_idx, val_idx = grouped_split(
        y.loc[trainval_idx], patients.loc[trainval_idx], n_splits=5, seed=seed
    )

    raw_train = X.loc[train_idx]
    expressed = raw_train.columns[raw_train.mean() >= min_mean_expr]
    variances = raw_train[expressed].var().sort_values(ascending=False)
    genes = list(variances.index[:n_genes])

    scaler = StandardScaler().fit(raw_train[genes])

    def scale(idx):
        return pd.DataFrame(scaler.transform(X.loc[idx, genes]), index=idx, columns=genes).astype(np.float64)

    return Dataset(
        X_train=scale(train_idx),
        X_val=scale(val_idx),
        X_test=scale(test_idx),
        y_train=y.loc[train_idx],
        y_val=y.loc[val_idx],
        y_test=y.loc[test_idx],
        genes=genes,
        scaler=scaler,
        raw_train=X.loc[train_idx, genes],
        raw_test=X.loc[test_idx, genes],
    )


def load_tcga_brca(cfg: dict, seed: int) -> Dataset:
    path = download(cfg["url"], cfg["path"])
    X, y, patients = label_samples(load_xena_matrix(path))
    return prepare(X, y, patients, cfg["n_genes"], cfg["min_mean_expr"], seed)


def make_synthetic(
    gene_pool: list[str],
    signal_genes: list[str],
    n_patients: int = 160,
    n_genes: int = 300,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Small fake TCGA-like cohort for tests / ``--demo``.

    Each patient contributes a tumor sample; ~30% also contribute a matched
    normal. ``signal_genes`` (which must be in ``gene_pool``) are shifted in
    tumors, so a correct pipeline should rank them highly and find them
    enriched.
    """
    rng = np.random.default_rng(seed)
    others = [g for g in gene_pool if g not in set(signal_genes)]
    rng.shuffle(others)
    genes = list(signal_genes) + others[: n_genes - len(signal_genes)]

    rows, labels, names = [], [], []
    for p in range(n_patients):
        patient_effect = rng.normal(0, 0.5, len(genes))
        pid = f"TCGA-SY-{p:04d}"
        samples = [(TUMOR_CODE, 1)] + ([(NORMAL_CODE, 0)] if rng.random() < 0.3 else [])
        for code, lab in samples:
            x = 6 + patient_effect + rng.normal(0, 1, len(genes))
            if lab == 1:
                x[: len(signal_genes)] += rng.normal(1.5, 0.5, len(signal_genes))
            rows.append(x)
            labels.append(lab)
            names.append(f"{pid}-{code}")
    X = pd.DataFrame(np.array(rows, dtype=np.float32), index=names, columns=genes)
    y = pd.Series(labels, index=names, name="tumor")
    patients = pd.Series(X.index.str[:12], index=names, name="patient")
    return X, y, patients
