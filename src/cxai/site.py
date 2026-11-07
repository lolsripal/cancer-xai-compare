"""Build the static GitHub Pages results site from the committed ``results/`` folder.

    python -m cxai.site --out _site

Nothing is recomputed: the site reads the CSV / JSON outputs of
``cxai.pipeline`` and ``cxai.stability``, embeds them as JSON, and renders
interactive charts with Plotly.js in the browser.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .site_template import TEMPLATE

ROOT = Path(__file__).resolve().parents[2]
REPO = "https://github.com/lolsripal/cancer-xai-compare"
TASKS = {"tumor_normal": "Tumor vs. normal", "subtype": "PAM50 subtype"}

# Interpretive sentences for each section. Numbers are filled in by the page from
# the results, so these stay true when the pipeline is re-run; keep claims here
# qualitative. Edit after looking at new results.
NARRATIVE: dict = {
    "tumor_normal": {
        "q1": "<strong>Mostly no.</strong> All four models are near-perfect (ROC-AUC about 0.999), yet they give very different reasons. "
              "Which model you explain changes the answer far more than which explainer you use.",
        "modules": "<strong>Partly.</strong> Grouping co-expressed genes lifts agreement between models, but the gap stays large. "
                   "The top module (hub gene CD300LG, 120 genes, Hallmark adipogenesis) is a fat-tissue program: the models largely detect the absence of normal breast tissue.",
        "q2": "<strong>Plausible, but mostly tissue composition.</strong> All consensus genes differ strongly between tumor and normal. About two-thirds are genes lost in tumors "
              "(fat and blood-vessel markers such as CA4 and ITIH5), and the rest are stromal or cell-cycle genes gained in tumors (COL10A1, MMP11, PKMYT1). "
              "PAM50 is not enriched, as expected: it was designed to separate subtypes, not tumor from normal. Apart from a likely spurious spermatogenesis hit for the neural network, only the random forest's rankings point to a pathway (G2M checkpoint, i.e. proliferation).",
        "q4": "<strong>No.</strong> Retraining without a model's own top 400 genes, without the largest gene module (120 genes), or without every PAM50 gene keeps ROC-AUC at 0.998 or above. "
              "The tumor/normal signal is spread over hundreds of interchangeable genes, so \"important\" does not mean \"necessary\".",
        "q5": "<strong>Yes.</strong> A logistic regression on 2 consensus genes reaches ROC-AUC 0.997, and on 5 genes it matches the full models. Depth-3 decision trees need about 20 genes.",
        "sanity": "Logistic regression, random forest and XGBoost <strong>pass</strong>: retrained on shuffled labels, their top genes overlap the real model's only at chance level. "
                  "The <strong>neural network partly fails</strong>: its shuffled-label explanation still shares 18% of the real top-50 (about 14&times; chance), "
                  "so its importance scores partly reflect the data rather than what it learned. "
                  "(Test AUC falls <em>below</em> 0.5 because, with shuffled labels, the small, distinctive cluster of normal samples ends up mostly labelled \"tumor\", so the models learn the reverse rule.)",
        "stability": "A single run's top-50 list is noisy: retrained on new patients, more than half of a model's list changes. Random forest is the most stable, and the neural network's SHAP list is no better than chance. Disagreement between models is still larger than that noise. "
                     "The 10 stable-core genes (COL10A1, MMP11, MME, PKMYT1, CES1, NEK2, CA4, FIGF, ATP1A2, TSLP) are a more trustworthy shortlist than any single run.",
    },
    "subtype": {
        "q1": "<strong>Still mostly no.</strong> On this harder 4-class task (balanced accuracy 0.86&ndash;0.89), gene-level agreement between models is only slightly higher than for tumor vs. normal.",
        "modules": "<strong>Here modules help more.</strong> The estrogen-response module (hub gene FOXA1, 48 genes) and the progesterone-receptor module (hub gene PGR) are in the top-50 modules for all 8 model &times; method combinations, "
                   "and removing whole top modules costs real accuracy, unlike removing the same number of random genes.",
        "q2": "<strong>Yes. This is the positive control, and the tree models pass it.</strong> Only 20 PAM50 genes survive the variance filter; random forest and XGBoost SHAP recover 11&ndash;12 of them in their top-50, "
              "logistic regression recovers 5, and the neural network recovers <strong>none</strong>. Top consensus genes include known luminal and basal markers (FOXA1, FOXC1, MAPT, GREB1, AGR2). "
              "Even though models disagree on individual genes, logistic regression, random forest and XGBoost all point to the <strong>same pathway</strong>, estrogen response, which is the core biology of luminal versus basal breast cancer.",
        "q4": "<strong>A little, unlike tumor vs. normal.</strong> Removing a model's top 50 genes costs up to 0.01 ROC-AUC, while removing random genes costs nothing. "
              "Removing the top 20 modules (80 genes) lowers random-forest balanced accuracy from 0.87 to 0.80, compared with 0.86 when 80 random genes are removed. "
              "Removing all 20 PAM50 genes, the genes the labels were defined from, only lowers balanced accuracy from about 0.87 to 0.83: the models rebuild the subtype signal from correlated genes.",
        "q5": "<strong>Nearly.</strong> Subtype needs more genes than tumor vs. normal. A logistic regression on 10 consensus genes reaches 0.86 balanced accuracy, and on 20 genes 0.87, "
              "within 0.02 of the best full model (0.89). Shallow trees lag further behind.",
        "sanity": "All models fall to roughly chance accuracy on shuffled labels (test ROC-AUC 0.45&ndash;0.63). Their explanations then overlap the real ones at about chance, "
                  "<strong>except the neural network</strong> (9% overlap, about 7&times; chance), the same warning sign as on the other task.",
        "stability": "The same pattern holds: single-run lists are noisy, the neural network's SHAP list is at chance, and different models disagree more than a model disagrees with itself across splits. "
                     "The stable core includes five PAM50 genes (ESR1, FOXA1, FOXC1, MAPT, MYBL2) and the estrogen target GREB1.",
    },
    "limitations": [
        "Tumor vs. normal is an easy task, and its normal samples are tumor-adjacent tissue (114 samples), not healthy breast; the differences partly reflect fat and stroma content.",
        "The PAM50 labels were computed from PAM50 gene expression, so the subtype task tests whether explainers recover a known rule, not new biology.",
        "Only 20&ndash;22 of the 50 PAM50 genes survive the top-2,000-variance filter; gene symbols in the 2016 Xena matrix are older than MSigDB 2024.",
        "Feature importance measures association with the prediction, not biological causation.",
        "Everything uses one cohort (TCGA); external validation (e.g. METABRIC) is the natural next step.",
        "MLP explanations rely on few SHAP samples and fail the label-randomization check, so treat them as the least reliable.",
    ],
}


def _csv(path: Path, **kw) -> pd.DataFrame | None:
    return pd.read_csv(path, **kw) if path.exists() else None


def _records(df: pd.DataFrame | None, digits: int = 4):
    if df is None:
        return []
    return json.loads(df.round(digits).to_json(orient="records"))


def task_payload(d: Path) -> dict | None:
    if not (d / "summary.json").exists():
        return None
    s = json.loads((d / "summary.json").read_text())
    k = s["top_k"]
    jac = pd.read_csv(d / f"jaccard_top{k}.csv", index_col=0)
    pairs = pd.read_csv(d / "pairwise_agreement.csv")
    by_type = pairs.groupby("comparison")[["jaccard_top_k", "x_chance"]].mean().reset_index()
    cons = pd.read_csv(d / "consensus_genes.csv").head(200)
    abl = pd.read_csv(d / "ablation_summary.csv")
    known = pd.read_csv(d / "known_gene_overlap.csv")
    enr = pd.read_csv(d / "pathway_enrichment_consensus.csv").head(12)
    enr["pathway"] = enr["pathway"].str.replace("HALLMARK_", "").str.replace("_", " ")
    pam = pd.read_csv(d / "ablation_pam50.csv")
    pam = pam.groupby(["model", "removed"])[["roc_auc", "balanced_accuracy"]].mean().reset_index()
    mods = _csv(d / "module_consensus.csv")
    figs = sorted(p.name for p in (d / "figures").glob("*.png"))
    return {
        "summary": s,
        "metrics": _records(pd.read_csv(d / "model_performance.csv")),
        "jaccard": {"labels": list(jac.columns), "z": jac.round(3).values.tolist()},
        "agreement_by_type": _records(by_type, 3),
        "module_vs_gene": _records(_csv(d / "module_vs_gene_agreement.csv"), 3),
        "known": _records(known[["ranking", "overlap", "precision_at_k", "p_value"]], 6),
        "enrichment": _records(enr[["pathway", "overlap", "list_size", "fold_enrichment", "fdr"]], 6),
        "ablation": _records(abl),
        "module_ablation": _records(_csv(d / "module_ablation.csv")),
        "pam50_removal": _records(pam),
        "sparse": _records(pd.read_csv(d / "sparse_models.csv").drop(columns="genes")),
        "sanity": _records(_csv(d / "sanity_label_randomization.csv")),
        "consensus": _records(cons),
        "modules": _records(mods.head(15)[["label", "size", "hallmark", "consensus_score", "genes"]]
                            if mods is not None else None),
        "figures": figs,
    }


def stability_payload(d: Path) -> dict | None:
    if not (d / "summary.json").exists():
        return None
    within = pd.read_csv(d / "ranking_stability.csv")
    cross = pd.read_csv(d / "cross_model_agreement_by_split.csv")
    cross_pairs = cross.groupby(["task", "a", "b"]).jaccard.mean().reset_index()
    freq = pd.read_csv(d / "gene_selection_frequency.csv")
    metrics = pd.read_csv(d / "metrics_by_split.csv")
    return {
        "summary": json.loads((d / "summary.json").read_text()),
        "within": _records(within, 3),
        "cross_pairs": _records(cross_pairs, 3),
        "freq": _records(freq.groupby("task").head(25), 2),
        "metrics": _records(metrics[["task", "seed", "model", "roc_auc", "balanced_accuracy"]]),
        "figures": sorted(p.name for p in (d / "figures").glob("*.png")),
    }


def build(results: Path = ROOT / "results", out: Path = ROOT / "_site") -> Path:
    results, out = Path(results), Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "figures").mkdir(parents=True)
    data = {"repo": REPO, "tasks": {}, "task_names": TASKS, "narrative": NARRATIVE}
    for t in TASKS:
        p = task_payload(results / t)
        if p is not None:
            data["tasks"][t] = p
            shutil.copytree(results / t / "figures", out / "figures" / t)
            shutil.copytree(results / t, out / "data" / t,
                            ignore=shutil.ignore_patterns("figures", "*.txt"))
    st = stability_payload(results / "stability")
    data["stability"] = st
    if st is not None:
        shutil.copytree(results / "stability" / "figures", out / "figures" / "stability")
        shutil.copytree(results / "stability", out / "data" / "stability", ignore=shutil.ignore_patterns("figures"))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    (out / "index.html").write_text(TEMPLATE.replace("__DATA__", payload).replace("__REPO__", REPO))
    (out / ".nojekyll").write_text("")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "results"))
    ap.add_argument("--out", default=str(ROOT / "_site"))
    args = ap.parse_args()
    print(f"Site written to {build(Path(args.results), Path(args.out)) / 'index.html'}")


if __name__ == "__main__":
    main()
