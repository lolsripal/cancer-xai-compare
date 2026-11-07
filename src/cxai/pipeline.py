"""End-to-end experiment: models -> predictions -> explanations -> comparisons.

    python -m cxai.pipeline --task all            # both TCGA-BRCA tasks (~15 min)
    python -m cxai.pipeline --task subtype        # one task
    python -m cxai.pipeline --task all --demo     # tiny synthetic cohorts, < 1 min
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

from . import ablation, biology, compare, explain, modules, plots, sanity, sparse
from .data import TASKS, load_task, make_synthetic, make_synthetic_subtypes, prepare
from .models import evaluate, fit_all

ROOT = Path(__file__).resolve().parents[2]


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_config(path: Path = ROOT / "config.yaml", demo: bool = False) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    if demo:
        cfg["explain"].update(top_k=20, shap_samples=40, mlp_shap_samples=8, perm_repeats=2, pdp_genes=3)
        cfg["ablation"].update(k_values=[5, 20], n_random=2)
        cfg["sparse"].update(k_values=[1, 3, 10])
        cfg["modules"].update(perm_repeats=1, ablation_m=[1, 3])
        cfg["stability"].update(seeds=[0, 1, 2], shap_samples=20, mlp_shap_samples=5)
    return cfg


def load_demo(task: str, seed: int):
    """Synthetic cohorts whose 'true' genes come from known Hallmark sets."""
    sets = biology.read_gmt()
    pool = sorted(set().union(*sets.values()))
    if task == "tumor_normal":
        signal = sorted(sets["HALLMARK_E2F_TARGETS"])[:20]
        X, y, p = make_synthetic(pool, signal, seed=seed)
        return prepare(X, y, p, n_genes=250, min_mean_expr=0.0, seed=seed), signal
    names = ["HALLMARK_E2F_TARGETS", "HALLMARK_ESTROGEN_RESPONSE_EARLY",
             "HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_HYPOXIA"]
    used, signals = set(), []
    for n in names:
        genes = [g for g in sorted(sets[n]) if g not in used][:8]
        used |= set(genes)
        signals.append(genes)
    X, y, p = make_synthetic_subtypes(pool, signals, seed=seed)
    data = prepare(X, y, p, n_genes=250, min_mean_expr=0.0, seed=seed,
                   class_names=["A", "B", "C", "D"])
    return data, [g for s in signals for g in s]


def get_data(task: str, cfg: dict, seed: int, demo: bool):
    if demo:
        return load_demo(task, seed)
    return load_task(task, cfg["data"], seed), None


def known_gene_set() -> set[str]:
    kg = biology.read_known_genes()
    return set(kg["gene"]) | set(kg["alias"].dropna())


def gene_annotations(cons: pd.DataFrame, data, g2p: dict, known: set[str]) -> pd.DataFrame:
    if data.binary:
        t = data.raw_train[data.y_train == 1].mean()
        n = data.raw_train[data.y_train == 0].mean()
        cons["log2FC_tumor_vs_normal"] = (t - n).loc[cons.index]
        cons["direction"] = np.where(cons["log2FC_tumor_vs_normal"] > 0, "up in tumor", "down in tumor")
    else:
        means = data.raw_train.groupby(data.y_train.values).mean()
        means.index = data.class_names
        cons["highest_in"] = means.idxmax().loc[cons.index]
        cons["lowest_in"] = means.idxmin().loc[cons.index]
        cons["log2_range_between_subtypes"] = (means.max() - means.min()).loc[cons.index]
    cons["PAM50"] = cons.index.isin(known)
    cons["hallmark_pathways"] = [";".join(sorted(g2p.get(g, []))) for g in cons.index]
    return cons


def agreement_by_type(imp: pd.DataFrame, k: int) -> pd.DataFrame:
    jac = compare.jaccard_matrix(imp, k)
    rho = compare.spearman_matrix(imp, k)
    return jac, rho, compare.pairwise_summary(jac, rho)


def run(task: str = "tumor_normal", demo: bool = False, config_path: Path = ROOT / "config.yaml",
        out: Path | None = None, models_dir: Path | None = None) -> dict:
    cfg = load_config(config_path, demo)
    seed = cfg["seed"]
    out = Path(out or (ROOT / "results" / ("demo" if demo else "") / task))
    models_dir = Path(models_dir or (ROOT / "models" / ("demo" if demo else "") / task))
    fig = out / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    k = cfg["explain"]["top_k"]
    title = {"tumor_normal": "tumor vs. normal", "subtype": "PAM50 subtype"}[task]

    # 1. Data ---------------------------------------------------------------
    log(f"=== Task: {task} ===")
    data, planted = get_data(task, cfg, seed, demo)
    log(f"  {data.summary()}")

    # 2. Models -> predictions ---------------------------------------------
    log("Training models")
    models = fit_all(data.X_train, data.y_train, seed, fast=demo)
    metrics = pd.DataFrame({n: evaluate(m, data.X_test, data.y_test) for n, m in models.items()}).T
    metrics.index.name = "model"
    metrics.to_csv(out / "model_performance.csv", float_format="%.4f")
    plots.performance(metrics, fig / "model_performance.png", f"Held-out test performance ({title})")
    log("\n" + metrics.round(4).to_string())

    # 3. Explanations -------------------------------------------------------
    log("Explaining models (validation split)")
    importance, shap_mats = explain.explain_all(models, data, cfg["explain"], seed, log)
    importance.index.name = "gene"
    importance.to_csv(out / "gene_importance.csv", float_format="%.6g")
    for name in ("xgboost", "logreg"):
        X_exp, sv = shap_mats[name]
        plots.shap_beeswarm(X_exp, sv, f"SHAP values: {name} ({title})",
                            fig / f"shap_summary_{name}.png", class_names=data.class_names)
    if not data.binary:
        per_class = pd.concat({n: explain.per_class_importance(sv, data.genes, data.class_names)
                               for n, (_, sv) in shap_mats.items()}, axis=1)
        tops = {f"{m}:{c}": list(per_class[(m, c)].sort_values(ascending=False).index[:10])
                for m, c in per_class.columns}
        pd.DataFrame(tops).to_csv(out / "top10_genes_per_class.csv", index_label="rank")

    # 4. Q1: agreement + chance baseline ------------------------------------
    log("Comparing rankings")
    jac, rho, pairs = agreement_by_type(importance, k)
    chance_gene = sanity.chance_jaccard(len(data.genes), k)
    pairs["x_chance"] = pairs["jaccard_top_k"] / chance_gene["mean"]
    jac.to_csv(out / f"jaccard_top{k}.csv", float_format="%.3f")
    rho.to_csv(out / f"spearman_union_top{k}.csv", float_format="%.3f")
    pairs.to_csv(out / "pairwise_agreement.csv", index=False, float_format="%.3f")
    plots.agreement_heatmap(jac, f"Jaccard overlap of top-{k} genes ({title})", fig / "agreement_jaccard.png")
    plots.agreement_heatmap(rho, f"Spearman rank correlation, union of top-{k} ({title})",
                            fig / "agreement_spearman.png")

    cons = compare.consensus(importance)
    gene_sets = biology.read_gmt()
    g2p = biology.gene_to_pathways(gene_sets)
    known = known_gene_set()
    cons = gene_annotations(cons, data, g2p, known)
    cons.index.name = "gene"
    cons.to_csv(out / "consensus_genes.csv", float_format="%.4f")
    consensus_rank = list(cons.index)
    plots.consensus_heatmap(compare.ranks(importance), consensus_rank[:30], fig / "consensus_top_genes.png")

    # 5. Q2 + Q3: biology ---------------------------------------------------
    log("Biological plausibility")
    tops = compare.top_k(importance, k)
    tops["consensus"] = consensus_rank[:k]
    known_df = pd.DataFrame([{"ranking": c, **biology.overlap_test(g, known, data.genes)}
                             for c, g in tops.items()])
    known_df["precision_at_k"] = known_df["overlap"] / known_df["list_size"]
    known_df["recall_of_pam50_in_features"] = known_df["overlap"] / known_df["set_size_in_background"]
    known_df.to_csv(out / "known_gene_overlap.csv", index=False, float_format="%.4g")

    enr = biology.enrichment(consensus_rank[:k], data.genes, gene_sets)
    enr.to_csv(out / "pathway_enrichment_consensus.csv", index=False, float_format="%.4g")
    plots.enrichment_bar(enr, fig / "pathway_enrichment.png")
    per_rank = [biology.enrichment(g, data.genes, gene_sets).assign(ranking=c) for c, g in tops.items()]
    per_rank_df = pd.concat([e for e in per_rank if not e.empty])
    per_rank_df[["ranking", "pathway", "overlap", "fdr"]].to_csv(
        out / "pathway_enrichment_by_ranking.csv", index=False, float_format="%.4g")

    # 6. Q4: ablation -------------------------------------------------------
    log("Ablation (retraining without top genes)")
    rankings = {n: list(importance[f"{n}:shap"].sort_values(ascending=False).index) for n in models}
    rankings["consensus"] = consensus_rank
    abl = ablation.ablate(models, rankings, data, cfg["ablation"]["k_values"],
                          cfg["ablation"]["n_random"], seed, log)
    abl.to_csv(out / "ablation_raw.csv", index=False, float_format="%.4f")
    abl_auc = ablation.summarise(abl, "roc_auc")
    abl_ll = ablation.summarise(abl, "log_loss")
    abl_auc.merge(abl_ll, on=["model", "removed", "k"]).to_csv(out / "ablation_summary.csv", index=False,
                                                                 float_format="%.4f")
    plots.ablation_curves(abl_auc, "roc_auc", fig / "ablation_auc.png")
    plots.ablation_curves(abl_ll, "log_loss", fig / "ablation_logloss.png")
    pam = ablation.ablate_gene_set(models, sorted(known), data, "pam50", cfg["ablation"]["n_random"], seed)
    pam.to_csv(out / "ablation_pam50.csv", index=False, float_format="%.4f")

    # 7. Q5: small interpretable models ------------------------------------
    log("Sparse interpretable models")
    depth = cfg["sparse"]["tree_depth"]
    sp, sp_models = sparse.run(consensus_rank, data, cfg["sparse"]["k_values"], depth, seed)
    sp.to_csv(out / "sparse_models.csv", index=False, float_format="%.4f")
    plots.sparse_curve(sp, metrics, fig / "sparse_vs_full.png")
    tree_k = 3 if 3 in cfg["sparse"]["k_values"] else cfg["sparse"]["k_values"][0]
    tg = consensus_rank[:tree_k]
    gi = [data.genes.index(g) for g in tg]
    rules = sparse.describe_tree(sp_models[(f"tree_depth{depth}", tree_k)], tg,
                                 data.scaler.mean_[gi], data.scaler.scale_[gi], data.class_names)
    (out / f"tree_rules_top{tree_k}.txt").write_text(rules + "\n")

    # 8. Gene modules -------------------------------------------------------
    log("Gene-module explanations")
    mods = modules.cluster_genes(data.X_train, cfg["modules"]["min_abs_corr"])
    mod_info = modules.describe_modules(mods, data.X_train, gene_sets, data.genes)
    mod_imp = modules.explain_modules(models, shap_mats, data, mods, cfg["modules"]["perm_repeats"], seed, log)
    k_mod = max(5, round(len(mod_imp) * k / len(data.genes)))
    mod_jac, mod_rho, mod_pairs = agreement_by_type(mod_imp, k_mod)
    chance_mod = sanity.chance_jaccard(len(mod_imp), k_mod)
    mod_pairs["x_chance"] = mod_pairs["jaccard_top_k"] / chance_mod["mean"]
    mod_pairs.to_csv(out / "module_pairwise_agreement.csv", index=False, float_format="%.3f")
    mod_cons = compare.consensus(mod_imp).join(mod_info)
    mod_cons.index.name = "module"
    mod_cons.to_csv(out / "module_consensus.csv", float_format="%.4f")
    plots.agreement_heatmap(mod_jac, f"Jaccard overlap of top-{k_mod} gene modules ({title})",
                            fig / "module_agreement_jaccard.png")

    gene_pairs = pairs.assign(level=f"genes (top {k} of {len(data.genes)})")
    mod_pairs_l = mod_pairs.assign(level=f"modules (top {k_mod} of {len(mod_imp)})")
    # Compare the *same* pairs of (model, method) at both levels.
    both = pd.concat([gene_pairs, mod_pairs_l])
    both = both[both.a.isin(mod_imp.columns) & both.b.isin(mod_imp.columns)]
    mvg = both.groupby(["level", "comparison"])[["jaccard_top_k", "x_chance"]].mean().reset_index()
    mvg.to_csv(out / "module_vs_gene_agreement.csv", index=False, float_format="%.3f")
    plots.module_vs_gene_agreement(both, fig / "module_vs_gene_agreement.png")

    mod_rank = list(mod_cons.index)
    mabl = modules.ablate_modules(models, mod_rank, mods, data, cfg["modules"]["ablation_m"],
                                  cfg["ablation"]["n_random"], seed, log)
    mabl_sum = (mabl.groupby(["model", "removed", "modules"])
                .agg(genes_removed=("genes_removed", "mean"), roc_auc=("roc_auc", "mean"),
                     log_loss=("log_loss", "mean"), balanced_accuracy=("balanced_accuracy", "mean"))
                .reset_index())
    mabl_sum.to_csv(out / "module_ablation.csv", index=False, float_format="%.4f")
    plots.module_ablation(mabl_sum, "balanced_accuracy", fig / "module_ablation.png")

    # 9. Sanity checks ------------------------------------------------------
    log("Sanity checks (label randomization)")
    lr = sanity.label_randomization(data, importance, cfg["explain"], k, seed, fast=demo, log=log)
    lr.to_csv(out / "sanity_label_randomization.csv", index=False, float_format="%.4f")

    # 10. Partial dependence ------------------------------------------------
    log("Partial dependence")
    pdp = explain.pdp_curves(models, data.X_val, consensus_rank[: cfg["explain"]["pdp_genes"]], data.class_names)
    pdp.to_csv(out / "partial_dependence.csv", index=False, float_format="%.4f")
    plots.pdp_grid(pdp, fig / "partial_dependence.png")

    # 11. Dashboard artefacts -----------------------------------------------
    rng = np.random.default_rng(seed)
    bg = data.X_train.iloc[rng.choice(len(data.X_train), size=min(200, len(data.X_train)), replace=False)]
    joblib.dump({
        "task": task, "class_names": data.class_names, "models": models, "scaler": data.scaler,
        "genes": data.genes, "background": bg, "gene_to_pathways": g2p, "known_genes": sorted(known),
        "consensus": cons.head(200), "metrics": metrics, "demo": demo,
    }, models_dir / "bundle.joblib", compress=3)
    ex = pd.concat([data.raw_test[data.y_test == c].head(2) for c in range(data.n_classes)])
    ex.insert(0, "true_label", [data.class_names[c] for c in data.y_test.loc[ex.index]])
    ex.index.name = "sample"
    ex.to_csv(out / "example_samples.csv", float_format="%.4f")

    # 12. Summary -----------------------------------------------------------
    by_kind = pairs.groupby("comparison")[["jaccard_top_k", "spearman_union_top_k", "x_chance"]].mean()
    pam_sum = pam.groupby("removed")[["roc_auc", "balanced_accuracy"]].mean()
    summary = {
        "task": task,
        "dataset": data.summary(),
        "top_k": k,
        "test_metrics": metrics.round(4).to_dict(orient="index"),
        "chance_jaccard_genes": chance_gene,
        "agreement_by_comparison_type": by_kind.round(3).to_dict(orient="index"),
        "consensus_top10": consensus_rank[:10],
        "known_gene_overlap": known_df.set_index("ranking")[["overlap", "precision_at_k", "p_value"]]
            .round(6).to_dict(orient="index"),
        "significant_pathways_consensus": enr[enr.fdr < 0.05][["pathway", "overlap", "fdr"]]
            .round(6).to_dict(orient="records"),
        "pam50_removal": {"n_pam50_genes_in_features": int(pam[pam.removed == "pam50"].k.iloc[0]),
                          **pam_sum.round(4).to_dict(orient="index")},
        "modules": {"n_modules": int(len(mod_imp)), "n_multi_gene": int((mod_info["size"] > 1).sum()),
                    "largest": int(mod_info["size"].max()), "top_k_modules": k_mod,
                    "chance_jaccard": chance_mod,
                    "module_vs_gene": mvg.round(3).to_dict(orient="records"),
                    "top5": mod_cons.head(5)[["label", "size", "hallmark"]].reset_index().to_dict(orient="records")},
        "sanity_label_randomization": lr.round(4).to_dict(orient="records"),
        "planted_signal_genes": planted,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"Done. Results in {out}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", choices=TASKS + ["all"])
    ap.add_argument("--demo", action="store_true", help="run on small synthetic datasets")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()
    for t in (TASKS if args.task == "all" else [args.task]):
        run(task=t, demo=args.demo, config_path=Path(args.config))


if __name__ == "__main__":
    main()
