"""Cancer AI Explainability Dashboard (research demo, not for clinical use).

    streamlit run app/dashboard.py

Requires a trained bundle from ``python -m cxai.pipeline`` (or ``--demo``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxai.explain import shap_values  # noqa: E402
from cxai.models import PRETTY  # noqa: E402

PATHWAY_NOTES = {
    "E2F_TARGETS": "Cell-cycle genes switched on by E2F transcription factors; a read-out of proliferation.",
    "G2M_CHECKPOINT": "Genes controlling entry into mitosis; high in rapidly dividing cells.",
    "MITOTIC_SPINDLE": "Machinery that separates chromosomes during cell division.",
    "MYC_TARGETS_V1": "Genes driven by the MYC oncogene (growth, ribosome biogenesis).",
    "MYC_TARGETS_V2": "A smaller MYC-driven program (nucleolar / biosynthesis genes).",
    "ESTROGEN_RESPONSE_EARLY": "Genes induced by estrogen receptor signalling, central to luminal breast cancer.",
    "ESTROGEN_RESPONSE_LATE": "Later estrogen-responsive genes; luminal breast cancer biology.",
    "EPITHELIAL_MESENCHYMAL_TRANSITION": "Stromal / mesenchymal genes linked to invasion and tissue remodelling.",
    "ADIPOGENESIS": "Fat-cell genes; normal breast tissue is adipose-rich, so these often mark NORMAL samples.",
    "FATTY_ACID_METABOLISM": "Lipid metabolism genes; frequently higher in fat-rich normal breast tissue.",
    "MYOGENESIS": "Muscle / contractile genes; can reflect myoepithelial or stromal content.",
    "ANGIOGENESIS": "Blood-vessel formation genes.",
    "KRAS_SIGNALING_DN": "Genes repressed by KRAS signalling.",
    "KRAS_SIGNALING_UP": "Genes induced by KRAS signalling.",
    "APICAL_JUNCTION": "Cell-cell adhesion genes that maintain epithelial structure.",
    "HYPOXIA": "Genes induced by low oxygen, common in solid tumours.",
    "GLYCOLYSIS": "Glucose metabolism genes; tumours often up-regulate glycolysis.",
    "DNA_REPAIR": "DNA damage repair machinery.",
    "P53_PATHWAY": "Targets of the p53 tumour suppressor.",
    "INFLAMMATORY_RESPONSE": "Immune / inflammatory genes reflecting immune infiltration.",
    "UV_RESPONSE_DN": "Genes down-regulated after UV stress; overlaps extracellular-matrix genes.",
    "COAGULATION": "Coagulation-cascade and protease genes, often stromal.",
}

st.set_page_config(page_title="Cancer AI Explainability Dashboard", layout="wide")


@st.cache_resource
def load_bundle():
    for p in (ROOT / "models" / "bundle.joblib", ROOT / "models" / "demo" / "bundle.joblib"):
        if p.exists():
            return joblib.load(p), p
    return None, None


@st.cache_data
def load_examples(demo: bool):
    p = ROOT / "results" / ("demo" if demo else "") / "example_samples.csv"
    return pd.read_csv(p, index_col=0) if p.exists() else None


bundle, bundle_path = load_bundle()
st.title("Cancer AI Explainability Dashboard")
st.caption("Breast tissue RNA-seq → prediction → confidence → top contributing genes → pathway interpretation. "
           "**Research demo only. Not for diagnosis or clinical decisions.**")

if bundle is None:
    st.error("No trained models found. Run `python -m cxai.pipeline` (or `--demo`) first.")
    st.stop()
if bundle["demo"]:
    st.warning("Using the synthetic **demo** bundle. Run the full pipeline for real TCGA-BRCA models.")

genes: list[str] = bundle["genes"]
scaler = bundle["scaler"]
models: dict = bundle["models"]
g2p: dict = bundle["gene_to_pathways"]
known = set(bundle["known_genes"])

# ---- input -------------------------------------------------------------
with st.sidebar:
    st.header("1. Sample")
    examples = load_examples(bundle["demo"])
    source = st.radio("Input", ["Example held-out sample", "Upload CSV"])
    raw = None
    true_label = None
    if source == "Upload CSV":
        st.markdown("CSV with gene symbols as columns and one row per sample; values in "
                    "log2(normalized count + 1), as in the UCSC Xena TCGA `HiSeqV2` matrix.")
        up = st.file_uploader("Expression CSV", type=["csv", "tsv", "txt"])
        if up is not None:
            sep = "\t" if up.name.endswith((".tsv", ".txt")) else ","
            df = pd.read_csv(up, sep=sep, index_col=0)
            if df.shape[0] > df.shape[1]:  # genes x samples -> transpose
                df = df.T
            choice = st.selectbox("Sample", df.index)
            raw = df.loc[choice]
    elif examples is not None:
        choice = st.selectbox("Sample", examples.index,
                              format_func=lambda s: f"{s} (true: {examples.loc[s, 'true_label']})")
        true_label = examples.loc[choice, "true_label"]
        raw = examples.loc[choice].drop("true_label")
    st.header("2. Explain with")
    explain_model = st.selectbox("Model", list(models), index=list(models).index("xgboost"),
                                 format_func=lambda m: PRETTY.get(m, m))
    n_top = st.slider("Genes to show", 5, 30, 15)

if raw is None:
    st.info("Choose an example sample or upload a CSV in the sidebar.")
    st.stop()

raw = pd.to_numeric(raw, errors="coerce")
missing = [g for g in genes if g not in raw.index or pd.isna(raw.get(g))]
vec = pd.Series(scaler.mean_, index=genes)
present = [g for g in genes if g not in set(missing)]
vec[present] = raw[present].astype(float)
if missing:
    st.warning(f"{len(missing)} of {len(genes)} model genes missing from the sample; "
               "they were set to the training mean (neutral).")
X = pd.DataFrame(scaler.transform(vec.to_frame().T), columns=genes)

# ---- prediction --------------------------------------------------------
probs = {m: float(models[m].predict_proba(X)[0, 1]) for m in models}
p = probs[explain_model]
label = "TUMOR" if p >= 0.5 else "NORMAL"
conf = p if p >= 0.5 else 1 - p

c1, c2, c3 = st.columns(3)
c1.metric(f"Prediction ({PRETTY[explain_model]})", label)
c2.metric("Confidence", f"{conf:.1%}")
c3.metric("Models agreeing", f"{sum((v >= 0.5) == (p >= 0.5) for v in probs.values())} / {len(probs)}")
if true_label:
    st.caption(f"Ground truth for this held-out sample: **{true_label}**")
st.dataframe(pd.DataFrame({"model": [PRETTY[m] for m in probs], "P(tumor)": list(probs.values())})
             .style.format({"P(tumor)": "{:.3f}"}), hide_index=True)

# ---- explanation -------------------------------------------------------
with st.spinner("Computing SHAP values..."):
    sv = shap_values(explain_model, models[explain_model], bundle["background"], X)[0]
contrib = pd.DataFrame({
    "gene": genes, "log2_expression": vec.values, "z_score": X.iloc[0].values, "shap": sv,
})
contrib["pushes_toward"] = np.where(contrib.shap > 0, "tumor", "normal")
contrib["PAM50"] = contrib.gene.isin(known)
contrib["hallmark_pathways"] = [", ".join(sorted(g2p.get(g, []))) for g in contrib.gene]
top = contrib.reindex(contrib.shap.abs().sort_values(ascending=False).index).head(n_top)

st.subheader("Top contributing genes")
st.caption("SHAP value = how much this gene's expression moved the model's log-odds of TUMOR for this sample, "
           "relative to an average training sample. Red pushes toward tumor, blue toward normal.")
chart = top[["gene", "shap"]].copy()
chart["direction"] = np.where(chart.shap > 0, "toward tumor", "toward normal")
st.vega_lite_chart(chart, {
    "mark": "bar",
    "encoding": {
        "y": {"field": "gene", "type": "nominal", "sort": "-x", "title": None},
        "x": {"field": "shap", "type": "quantitative", "title": "SHAP value"},
        "color": {"field": "direction", "type": "nominal",
                  "scale": {"domain": ["toward tumor", "toward normal"], "range": ["#C44E52", "#4C72B0"]}},
    },
}, width="stretch")
st.dataframe(top.style.format({"log2_expression": "{:.2f}", "z_score": "{:+.2f}", "shap": "{:+.3f}"}),
             hide_index=True, width="stretch")

# ---- pathway interpretation -------------------------------------------
st.subheader("Biological pathway interpretation")
pool = contrib.reindex(contrib.shap.abs().sort_values(ascending=False).index).head(50)
rows = []
for _, r in pool.iterrows():
    for pw in g2p.get(r.gene, []):
        rows.append({"pathway": pw, "gene": r.gene, "shap": r.shap})
if rows:
    pw = (pd.DataFrame(rows).groupby("pathway")
          .agg(total_abs_shap=("shap", lambda s: s.abs().sum()), net_shap=("shap", "sum"),
               genes=("gene", lambda g: ", ".join(g)))
          .sort_values("total_abs_shap", ascending=False).head(8).reset_index())
    for _, r in pw.iterrows():
        toward = "tumor" if r.net_shap > 0 else "normal"
        note = PATHWAY_NOTES.get(r.pathway, "")
        st.markdown(f"**{r.pathway.replace('_', ' ').title()}**: net push toward *{toward}* "
                    f"(Σ|SHAP| = {r.total_abs_shap:.2f}). Genes: {r.genes}.  \n{note}")
else:
    st.write("None of the top 50 contributing genes belong to a Hallmark gene set.")

with st.expander("What the study found across the whole cohort"):
    st.markdown("Global consensus genes (averaged over 4 models × 3 explanation methods):")
    st.dataframe(bundle["consensus"].head(25), width="stretch")
    st.markdown("Test-set performance of each model:")
    st.dataframe(bundle["metrics"].round(4), width="stretch")
st.caption(f"Model bundle: `{bundle_path.relative_to(ROOT)}`")
