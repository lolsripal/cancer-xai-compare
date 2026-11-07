import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from cxai import biology, compare
from cxai.data import grouped_split, label_samples, make_synthetic, prepare


@pytest.fixture(scope="module")
def synthetic():
    pool = [f"G{i}" for i in range(400)]
    signal = pool[:10]
    X, y, patients = make_synthetic(pool, signal, n_patients=120, n_genes=150, seed=1)
    return X, y, patients, signal


def test_label_samples_from_barcodes():
    idx = ["TCGA-AA-0001-01", "TCGA-AA-0001-11", "TCGA-AA-0002-06", "TCGA-AA-0003-01"]
    expr = pd.DataFrame(np.ones((4, 2)), index=idx, columns=["A", "B"])
    X, y, patients = label_samples(expr)
    assert list(X.index) == ["TCGA-AA-0001-01", "TCGA-AA-0001-11", "TCGA-AA-0003-01"]
    assert list(y) == [1, 0, 1]  # metastatic (-06) samples are dropped
    assert list(patients) == ["TCGA-AA-0001", "TCGA-AA-0001", "TCGA-AA-0003"]


def test_splits_never_share_a_patient(synthetic):
    X, y, patients, _ = synthetic
    data = prepare(X, y, patients, n_genes=100, min_mean_expr=0.0, seed=0)
    sets = [set(patients.loc[d.index]) for d in (data.X_train, data.X_val, data.X_test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    for part in (data.y_train, data.y_val, data.y_test):
        assert part.nunique() == 2  # both classes present in every split
    assert np.allclose(data.X_train.mean(), 0, atol=1e-6)  # scaler fitted on train only


def test_grouped_split_is_deterministic(synthetic):
    _, y, patients, _ = synthetic
    a = grouped_split(y, patients, 5, seed=3)
    b = grouped_split(y, patients, 5, seed=3)
    assert list(a[1]) == list(b[1])


def test_overlap_test_matches_hypergeometric():
    background = [f"G{i}" for i in range(100)]
    reference = set(background[:20])
    hits = background[:5] + background[50:55]  # 5 of 10 hits are in the reference
    r = biology.overlap_test(hits, reference, background)
    assert r["overlap"] == 5 and r["list_size"] == 10 and r["set_size_in_background"] == 20
    assert r["p_value"] == pytest.approx(hypergeom.sf(4, 100, 20, 10))
    assert r["fold_enrichment"] == pytest.approx(5 / (10 * 20 / 100))


def test_enrichment_applies_fdr():
    background = [f"G{i}" for i in range(200)]
    sets = {"A": set(background[:20]), "B": set(background[100:120]), "tiny": {"G0"}}
    df = biology.enrichment(background[:15], background, sets)
    assert list(df.pathway) == ["A", "B"]  # "tiny" is below min_size
    assert df.loc[0, "fdr"] < 0.05 and df.loc[1, "fdr"] == pytest.approx(1.0)


def test_rank_agreement_and_consensus():
    imp = pd.DataFrame({
        "m1:shap": [5, 4, 3, 2, 1],
        "m2:shap": [5, 4, 3, 2, 1],
        "m3:perm": [1, 2, 3, 4, 5],
    }, index=list("abcde"))
    jac = compare.jaccard_matrix(imp, k=2)
    assert jac.loc["m1:shap", "m2:shap"] == 1.0 and jac.loc["m1:shap", "m3:perm"] == 0.0
    assert np.allclose(jac.values, jac.values.T)
    rho = compare.spearman_matrix(imp, k=2)
    assert rho.loc["m1:shap", "m2:shap"] == pytest.approx(1.0)
    assert rho.loc["m1:shap", "m3:perm"] < 0
    cons = compare.consensus(imp)
    assert cons.index[0] in {"a", "b"}


def test_gmt_and_known_genes_load():
    sets = biology.read_gmt()
    assert len(sets) == 50 and "HALLMARK_E2F_TARGETS" in sets
    known = biology.read_known_genes()
    assert len(known) == 50 and {"ESR1", "ERBB2", "MKI67"} <= set(known.gene)


def test_demo_pipeline_recovers_planted_signal(tmp_path):
    from cxai.pipeline import run

    summary = run(demo=True, out=tmp_path / "results", models_dir=tmp_path / "models")
    planted = set(summary["planted_signal_genes"])
    # The consensus ranking should be dominated by the genes we shifted in tumors...
    assert len(planted & set(summary["consensus_top10"])) >= 8
    # ...and enrichment should point at the pathway those genes came from.
    sig = {p["pathway"] for p in summary["significant_pathways_consensus"]}
    assert "HALLMARK_E2F_TARGETS" in sig
    assert (tmp_path / "models" / "bundle.joblib").exists()
    assert (tmp_path / "results" / "figures" / "ablation_auc.png").exists()
