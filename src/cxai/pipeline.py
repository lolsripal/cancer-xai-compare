"""End-to-end experiment: models -> predictions -> explanations -> comparisons.

    python -m cxai.pipeline            # full TCGA-BRCA run (downloads ~64 MB once)
    python -m cxai.pipeline --demo     # tiny synthetic cohort, runs in under a minute
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from . import ablation, biology, compare, explain, plots, sparse
from .data import load_tcga_brca, make_synthetic, prepare
from .models import evaluate, fit_all

ROOT = Path(__file__).resolve().parents[2]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def demo_config(cfg: dict) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["explain"].update(top_k=20, shap_samples=40, mlp_shap_samples=8, perm_repeats=2, pdp_genes=3)
    cfg["ablation"].update(k_values=[5, 20], n_random=2)
    cfg["sparse"].update(k_values=[1, 3, 10])
    return cfg


def load_demo(seed: int):
    sets = biology.read_gmt()
    pool = sorted(set().union(*sets.values()))
    signal = sorted(sets["HALLMARK_E2F_TARGETS"])[:20]
    X, y, patients = make_synthetic(pool, signal, seed=seed)
    return prepare(X, y, patients, n_genes=250, min_mean_expr=0.0, seed=seed), signal


def known_gene_set() -> set[str]:
    kg = biology.read_known_genes()
    return set(kg["gene"]) | set(kg["alias"].dropna())


def run(demo: bool = False, config_path: Path = ROOT / "config.yaml", out: Path | None = None,
        models_dir: Path | None = None) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = cfg["seed"]
    if demo:
        cfg = demo_config(cfg)
    out = Path(out or (ROOT / "results" / ("demo" if demo else "")))
    models_dir = Path(models_dir or (ROOT / "models" / ("demo" if demo else "")))
    (out / "figures").mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    k = cfg["explain"]["top_k"]

    # 1. Data ---------------------------------------------------------------
    log("Loading data")
    if demo:
        data, planted = load_demo(seed)
    else:
        data, planted = load_tcga_brca(cfg["data"], seed), None
    log(f"  {data.summary()}")

    # 2. Models -> predictions ---------------------------------------------
    log("Training models")
    models = fit_all(data.X_train, data.y_train, seed, fast=demo)
    metrics = pd.DataFrame({n: evaluate(m, data.X_test, data.y_test) for n, m in models.items()}).T
    metrics.index.name = "model"
    metrics.to_csv(out / "model_performance.csv", float_format="%.4f")
    plots.performance(metrics, out / "figures" / "model_performance.png")
    log("\n" + metrics.round(4).to_string())

    # 3. Explanations -------------------------------------------------------
    log("Explaining models (validation split)")
    importance, shap_mats = explain.explain_all(models, data, cfg["explain"], seed, log)
    importance.index.name = "gene"
    importance.to_csv(out / "gene_importance.csv", float_format="%.6g")
    for name in ("xgboost", "logreg"):
        X_exp, sv = shap_mats[name]
        plots.shap_beeswarm(X_exp, sv, f"SHAP values: {name}", out / "figures" / f"shap_beeswarm_{name}.png")

    # 4. Q1: do models agree? ---------------------------------------------
    log("Comparing rankings")
    jac = compare.jaccard_matrix(importance, k)
    rho = compare.spearman_matrix(importance, k)
    jac.to_csv(out / f"jaccard_top{k}.csv", float_format="%.3f")
    rho.to_csv(out / f"spearman_union_top{k}.csv", float_format="%.3f")
    pairs = compare.pairwise_summary(jac, rho)
    pairs.to_csv(out / "pairwise_agreement.csv", index=False, float_format="%.3f")
    plots.agreement_heatmap(jac, f"Jaccard overlap of top-{k} genes", out / "figures" / "agreement_jaccard.png")
    plots.agreement_heatmap(rho, f"Spearman rank correlation (union of top-{k} lists)",
                            out / "figures" / "agreement_spearman.png")

    cons = compare.consensus(importance)
    gene_sets = biology.read_gmt()
    g2p = biology.gene_to_pathways(gene_sets)
    known = known_gene_set()
    tumor_mean = data.raw_train[data.y_train == 1].mean()
    normal_mean = data.raw_train[data.y_train == 0].mean()
    cons["log2FC_tumor_vs_normal"] = (tumor_mean - normal_mean).loc[cons.index]
    cons["direction"] = np.where(cons["log2FC_tumor_vs_normal"] > 0, "up in tumor", "down in tumor")
    cons["PAM50"] = cons.index.isin(known)
    cons["hallmark_pathways"] = [";".join(sorted(g2p.get(g, []))) for g in cons.index]
    cons.index.name = "gene"
    cons.to_csv(out / "consensus_genes.csv", float_format="%.4f")
    consensus_rank = list(cons.index)
    plots.consensus_heatmap(compare.ranks(importance), consensus_rank[:30],
                            out / "figures" / "consensus_top_genes.png")

    # 5. Q2 + Q3: biological plausibility -----------------------------------
    log("Biological plausibility")
    tops = compare.top_k(importance, k)
    tops["consensus"] = consensus_rank[:k]
    known_rows = [{"ranking": c, **biology.overlap_test(g, known, data.genes)} for c, g in tops.items()]
    known_df = pd.DataFrame(known_rows)
    known_df.to_csv(out / "known_gene_overlap.csv", index=False, float_format="%.4g")

    enr = biology.enrichment(consensus_rank[:k], data.genes, gene_sets)
    enr.to_csv(out / "pathway_enrichment_consensus.csv", index=False, float_format="%.4g")
    plots.enrichment_bar(enr, out / "figures" / "pathway_enrichment.png")
    per_model = []
    for c, g in tops.items():
        e = biology.enrichment(g, data.genes, gene_sets)
        if not e.empty:
            per_model.append(e.assign(ranking=c)[["ranking", "pathway", "overlap", "fdr"]])
    per_model_df = pd.concat(per_model)
    per_model_df.to_csv(out / "pathway_enrichment_by_ranking.csv", index=False, float_format="%.4g")
    coverage = {name: len(s & set(data.genes)) for name, s in gene_sets.items()}

    # 6. Q4: ablation -------------------------------------------------------
    log("Ablation (retraining without top genes)")
    rankings = {n: list(importance[f"{n}:shap"].sort_values(ascending=False).index) for n in models}
    rankings["consensus"] = consensus_rank
    abl = ablation.ablate(models, rankings, data, cfg["ablation"]["k_values"],
                          cfg["ablation"]["n_random"], seed, log)
    abl.to_csv(out / "ablation_raw.csv", index=False, float_format="%.4f")
    abl_auc = ablation.summarise(abl, "roc_auc")
    abl_ll = ablation.summarise(abl, "log_loss")
    abl_auc.merge(abl_ll, on=["model", "removed", "k"]).to_csv(
        out / "ablation_summary.csv", index=False, float_format="%.4f")
    plots.ablation_curves(abl_auc, "roc_auc", out / "figures" / "ablation_auc.png")
    plots.ablation_curves(abl_ll, "log_loss", out / "figures" / "ablation_logloss.png")

    # 7. Q5: small interpretable models ------------------------------------
    log("Sparse interpretable models")
    sp, sp_models = sparse.run(consensus_rank, data, cfg["sparse"]["k_values"],
                               cfg["sparse"]["tree_depth"], seed)
    sp.to_csv(out / "sparse_models.csv", index=False, float_format="%.4f")
    plots.sparse_curve(sp, metrics, out / "figures" / "sparse_vs_full.png")
    depth = cfg["sparse"]["tree_depth"]
    tree_k = 3 if 3 in cfg["sparse"]["k_values"] else cfg["sparse"]["k_values"][0]
    tree_genes = consensus_rank[:tree_k]
    gi = [data.genes.index(g) for g in tree_genes]
    rules = sparse.describe_tree(sp_models[(f"tree_depth{depth}", tree_k)], tree_genes,
                                 data.scaler.mean_[gi], data.scaler.scale_[gi])
    (out / f"tree_rules_top{tree_k}.txt").write_text(rules + "\n")

    # 8. Partial dependence -------------------------------------------------
    log("Partial dependence")
    pdp_genes = consensus_rank[: cfg["explain"]["pdp_genes"]]
    pdp = explain.pdp_curves(models, data.X_val, pdp_genes)
    pdp.to_csv(out / "partial_dependence.csv", index=False, float_format="%.4f")
    plots.pdp_grid(pdp, out / "figures" / "partial_dependence.png")

    # 9. Artefacts for the dashboard ---------------------------------------
    rng = np.random.default_rng(seed)
    bg = data.X_train.iloc[rng.choice(len(data.X_train), size=min(200, len(data.X_train)), replace=False)]
    joblib.dump({
        "models": models, "scaler": data.scaler, "genes": data.genes, "background": bg,
        "gene_to_pathways": g2p, "known_genes": sorted(known),
        "consensus": cons.head(200), "metrics": metrics, "demo": demo,
    }, models_dir / "bundle.joblib", compress=3)
    ex_t = data.raw_test[data.y_test == 1].head(3)
    ex_n = data.raw_test[data.y_test == 0].head(2)
    examples = pd.concat([ex_t, ex_n])
    examples.insert(0, "true_label", ["tumor"] * len(ex_t) + ["normal"] * len(ex_n))
    examples.index.name = "sample"
    examples.to_csv(out / "example_samples.csv", float_format="%.4f")

    # 10. Summary -----------------------------------------------------------
    by_kind = pairs.groupby("comparison")[["jaccard_top_k", "spearman_union_top_k"]].mean()
    summary = {
        "dataset": data.summary(),
        "top_k": k,
        "test_metrics": metrics.round(4).to_dict(orient="index"),
        "agreement_by_comparison_type": by_kind.round(3).to_dict(orient="index"),
        "consensus_top10": consensus_rank[:10],
        "known_gene_overlap_consensus": known_df[known_df.ranking == "consensus"].iloc[0].to_dict(),
        "significant_pathways_consensus": enr[enr.fdr < 0.05][["pathway", "overlap", "fdr"]]
            .round(6).to_dict(orient="records"),
        "hallmark_genes_in_feature_space": int(sum(coverage.values())),
        "planted_signal_genes": planted,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"Done. Results in {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="run on a small synthetic dataset")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    run(demo=args.demo, config_path=Path(args.config))


if __name__ == "__main__":
    main()
