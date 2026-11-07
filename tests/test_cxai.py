import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from cxai import biology, compare, modules, sanity
from cxai.data import (grouped_split, label_samples, label_subtypes, make_synthetic,
                       make_synthetic_subtypes, prepare)
from cxai.models import evaluate, fit_all


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

    summary = run(task="tumor_normal", demo=True, out=tmp_path / "results", models_dir=tmp_path / "models")
    planted = set(summary["planted_signal_genes"])
    # The consensus ranking should be dominated by the genes we shifted in tumors...
    assert len(planted & set(summary["consensus_top10"])) >= 8
    # ...and enrichment should point at the pathway those genes came from.
    sig = {p["pathway"] for p in summary["significant_pathways_consensus"]}
    assert "HALLMARK_E2F_TARGETS" in sig
    assert (tmp_path / "models" / "bundle.joblib").exists()
    assert (tmp_path / "results" / "figures" / "ablation_auc.png").exists()


def test_label_subtypes_keeps_primary_tumors_with_pam50_call():
    idx = ["TCGA-AA-0001-01", "TCGA-AA-0002-01", "TCGA-AA-0003-11", "TCGA-AA-0004-01"]
    expr = pd.DataFrame(np.ones((4, 2)), index=idx, columns=["A", "B"])
    clin = pd.DataFrame({"PAM50Call_RNAseq": ["Basal", "Normal", "LumA", "LumB"]}, index=idx)
    X, y, _ = label_subtypes(expr, clin)
    assert list(X.index) == ["TCGA-AA-0001-01", "TCGA-AA-0004-01"]  # Normal-like and -11 dropped
    assert list(y) == [3, 1]  # SUBTYPES = LumA, LumB, Her2, Basal


def test_multiclass_models_and_metrics():
    pool = [f"G{i}" for i in range(300)]
    signals = [pool[i * 5:(i + 1) * 5] for i in range(4)]
    X, y, p = make_synthetic_subtypes(pool, signals, n_patients=200, n_genes=80, seed=0)
    data = prepare(X, y, p, n_genes=60, min_mean_expr=0.0, seed=0, class_names=list("ABCD"))
    models = fit_all(data.X_train, data.y_train, fast=True)
    for m in models.values():
        r = evaluate(m, data.X_test, data.y_test)
        assert set(r) >= {"roc_auc", "balanced_accuracy", "f1", "log_loss"}
        assert r["roc_auc"] > 0.8


def test_chance_jaccard_matches_expectation():
    c = sanity.chance_jaccard(2000, 50)
    # E[overlap] = k^2 / n = 1.25, so Jaccard is about 1.25 / 98.75
    assert c["mean"] == pytest.approx(1.25 / 98.75, rel=0.1)
    assert c["p95"] >= c["mean"]


def test_cluster_genes_groups_correlated_genes():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(200, 2))
    cols = {f"a{i}": base[:, 0] + rng.normal(0, 0.2, 200) for i in range(5)}
    cols |= {f"b{i}": base[:, 1] + rng.normal(0, 0.2, 200) for i in range(4)}
    cols |= {f"n{i}": rng.normal(size=200) for i in range(3)}
    mods = modules.cluster_genes(pd.DataFrame(cols), min_abs_corr=0.6)
    assert mods[[f"a{i}" for i in range(5)]].nunique() == 1
    assert mods[[f"b{i}" for i in range(4)]].nunique() == 1
    assert mods["a0"] != mods["b0"]
    assert mods.value_counts().iloc[0] == 5 and mods.loc["a0"] == 1  # largest module is M1


def test_module_shap_is_sum_within_module():
    values = np.array([[[1.0], [-1.0], [2.0]], [[0.5], [0.5], [-1.0]]])  # (2 samples, 3 genes, 1 class)
    mods = pd.Series([1, 1, 2], index=["g1", "g2", "g3"])
    imp = modules.module_shap(values, ["g1", "g2", "g3"], mods)
    assert imp[1] == pytest.approx((0 + 1.0) / 2)  # opposite signs cancel in sample 1
    assert imp[2] == pytest.approx((2 + 1) / 2)


def test_demo_subtype_pipeline_and_stability(tmp_path):
    from cxai import stability
    from cxai.pipeline import run

    summary = run(task="subtype", demo=True, out=tmp_path / "r", models_dir=tmp_path / "m")
    assert summary["dataset"]["classes"] == list("ABCD")
    assert len(set(summary["planted_signal_genes"]) & set(summary["consensus_top10"])) >= 7
    for row in summary["sanity_label_randomization"]:
        assert row["test_roc_auc_random_labels"] < 0.75  # shuffled labels -> near chance
    st = stability.run(["subtype"], demo=True, out=tmp_path / "s")
    assert 0 <= st["subtype"]["within_method_stability_mean"] <= 1


def test_site_builds_from_demo_results(tmp_path):
    from cxai import site
    from cxai.pipeline import run

    res = tmp_path / "results"
    for t in ("tumor_normal", "subtype"):
        run(task=t, demo=True, out=res / t, models_dir=tmp_path / "m" / t)
    out = site.build(results=res, out=tmp_path / "_site")
    html = (out / "index.html").read_text()
    assert "<title>" in html and "Plotly" in html and "Gene explorer" in html
    assert (out / "figures" / "tumor_normal" / "agreement_jaccard.png").exists()
