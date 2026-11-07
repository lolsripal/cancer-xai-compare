"""HTML template for the GitHub Pages results site (filled in by ``cxai.site``)."""

TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cancer XAI Compare</title>
<meta name="description" content="Do different machine-learning models agree on which genes drive breast cancer predictions? A comparative explainability study on TCGA-BRCA.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
:root {
  color-scheme: light;
  --bg: #f6f5f2; --surface-1: #fcfcfb; --surface-2: #f0efec; --border: #e2e0da;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #7a7974;
  --accent: #2a78d6; --accent-soft: #cde2fb;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100;
  --grid: #e8e6e1; --ref: #8a8984;
  --seq-0: #f0efec; --seq-1: #b7d3f6; --seq-2: #6da7ec; --seq-3: #2a78d6; --seq-4: #184f95; --seq-5: #0d366b;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --bg: #121211; --surface-1: #1a1a19; --surface-2: #232322; --border: #33332f;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --accent: #3987e5; --accent-soft: #1c3557;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
    --grid: #2c2c29; --ref: #8f8e86;
    --seq-0: #232322; --seq-1: #104281; --seq-2: #1c5cab; --seq-3: #3987e5; --seq-4: #86b6ef; --seq-5: #cde2fb;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #121211; --surface-1: #1a1a19; --surface-2: #232322; --border: #33332f;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
  --accent: #3987e5; --accent-soft: #1c3557;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500;
  --grid: #2c2c29; --ref: #8f8e86;
  --seq-0: #232322; --seq-1: #104281; --seq-2: #1c5cab; --seq-3: #3987e5; --seq-4: #86b6ef; --seq-5: #cde2fb;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
html, body { overflow-wrap: anywhere; }
body { margin: 0; background: var(--bg); color: var(--text-primary);
  font: 16px/1.6 Inter, system-ui, -apple-system, "Segoe UI", sans-serif; -webkit-font-smoothing: antialiased; }
a { color: var(--accent); text-decoration: none; } a:hover { text-decoration: underline; }
code, .mono { font-family: "JetBrains Mono", ui-monospace, monospace; font-size: .88em; }
.wrap { max-width: 1120px; margin: 0 auto; padding: 0 16px; }
header.top { border-bottom: 1px solid var(--border); background: var(--surface-1); }
header.top .wrap { display: flex; align-items: center; gap: 8px 16px; min-height: 56px; flex-wrap: wrap; padding-top: 8px; padding-bottom: 8px; }
header.top .brand { font-weight: 700; letter-spacing: -.01em; }
header.top nav { margin-left: auto; display: flex; gap: 14px; align-items: center; font-size: 14px; flex-wrap: wrap; }
header.top nav a { color: var(--text-secondary); }
.btn-icon { background: none; border: 1px solid var(--border); color: var(--text-secondary); border-radius: 8px;
  height: 32px; padding: 0 10px; cursor: pointer; font: inherit; font-size: 13px; }
.hero { padding-top: 56px; padding-bottom: 28px; }
.eyebrow { text-transform: uppercase; letter-spacing: .08em; font-size: 12px; font-weight: 600; color: var(--accent); }
h1 { font-size: clamp(30px, 5vw, 46px); line-height: 1.1; letter-spacing: -.025em; margin: 10px 0 16px; max-width: 860px; }
.lede { font-size: 18px; color: var(--text-secondary); max-width: 760px; margin: 0; }
.taskbar { position: sticky; top: 0; z-index: 10; background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); }
.taskbar .wrap { display: flex; align-items: center; gap: 12px; padding-top: 10px; padding-bottom: 10px; flex-wrap: wrap; }
.seg { display: inline-flex; background: var(--surface-2); border-radius: 10px; padding: 3px; }
.seg button { border: 0; background: none; padding: 7px 14px; border-radius: 8px; font: inherit; font-size: 14px;
  font-weight: 500; color: var(--text-secondary); cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--surface-1); color: var(--text-primary); box-shadow: 0 1px 2px rgba(0,0,0,.12); }
.taskbar .hint { font-size: 13px; color: var(--text-muted); }
.toc { display: flex; gap: 14px; flex-wrap: wrap; font-size: 13px; margin-left: auto; }
.toc a { color: var(--text-secondary); }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(220px, 100%), 1fr)); gap: 12px; margin: 28px 0 8px; }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 14px; padding: 18px; }
.tile .k { font-size: 13px; color: var(--text-secondary); }
.tile .v { font-size: 30px; font-weight: 700; letter-spacing: -.02em; margin: 4px 0 2px; font-variant-numeric: tabular-nums; }
.tile .s { font-size: 13px; color: var(--text-muted); }
section.q { padding: 40px 0 8px; border-top: 1px solid var(--border); margin-top: 32px; }
section.q h2 { font-size: 26px; letter-spacing: -.015em; margin: 6px 0 10px; }
section.q .qn { font-size: 13px; font-weight: 600; color: var(--accent); }
.answer { font-size: 17px; max-width: 820px; margin: 0 0 6px; }
.answer strong { font-weight: 650; }
.note { color: var(--text-secondary); max-width: 820px; font-size: 15px; }
.grid2 { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 16px; margin-top: 18px; }
@media (max-width: 860px) { .grid2 { grid-template-columns: minmax(0, 1fr); } .toc { display: none; } }
@media (max-width: 520px) { .chart.tall { height: 400px; } section.q h2 { font-size: 22px; } }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 14px; padding: 16px; min-width: 0; }
.card h3 { font-size: 15px; margin: 0 0 2px; }
.card .sub { font-size: 13px; color: var(--text-muted); margin: 0 0 8px; }
.chart { width: 100%; height: 360px; }
.chart.tall { height: 460px; }
.card .foot { font-size: 12px; color: var(--text-muted); margin-top: 6px; display: flex; gap: 12px; flex-wrap: wrap; }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-size: 13px; color: var(--text-secondary); margin-bottom: 6px; }
select, input[type=search] { font: inherit; font-size: 14px; color: var(--text-primary); background: var(--surface-2);
  border: 1px solid var(--border); border-radius: 8px; padding: 6px 10px; }
input[type=search] { width: min(320px, 100%); }
table { width: 100%; border-collapse: collapse; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { font-size: 12px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-secondary); font-weight: 600;
  cursor: pointer; user-select: none; position: sticky; top: 0; background: var(--surface-1); }
td.wrap-cell { white-space: normal; min-width: 220px; color: var(--text-secondary); font-size: 13px; }
.table-scroll { overflow: auto; max-height: 520px; border-radius: 10px; }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; background: var(--accent-soft); color: var(--text-primary); font-size: 12px; }
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(240px, 100%), 1fr)); gap: 12px; margin-top: 16px; }
.gallery a { display: block; background: #fff; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
.gallery img { width: 100%; height: 160px; object-fit: contain; display: block; }
.gallery span { display: block; font-size: 12px; color: #52514e; padding: 6px 8px; border-top: 1px solid #e2e0da; background: #fcfcfb; }
ul.clean { padding-left: 18px; max-width: 820px; } ul.clean li { margin: 6px 0; color: var(--text-secondary); }
ul.clean li strong { color: var(--text-primary); }
pre { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 14px; overflow: auto; font-size: 13px; }
footer { border-top: 1px solid var(--border); margin-top: 48px; padding: 24px 0 40px; color: var(--text-muted); font-size: 13px; }
.empty { color: var(--text-muted); font-size: 14px; padding: 20px 0; }
</style>
</head>
<body>
<header class="top"><div class="wrap">
  <span class="brand">cancer-xai-compare</span>
  <nav>
    <a href="#methods">Methods</a>
    <a href="#explorer">Gene explorer</a>
    <a href="__REPO__">GitHub</a>
    <button class="btn-icon" id="theme" aria-label="Toggle dark mode">Theme</button>
  </nav>
</div></header>

<div class="wrap hero">
  <div class="eyebrow">Explainable AI · TCGA breast cancer RNA-seq</div>
  <h1>When an ML model calls a tumor malignant, do different models agree on <em>why</em>?</h1>
  <p class="lede">Four models (logistic regression, random forest, XGBoost and a neural network) were trained on 1,211 breast tissue samples, and each was explained three ways (SHAP, permutation importance, built-in importance). The comparison is run on two tasks: an easy one (<strong>tumor vs. normal</strong>) and a harder one with a known answer (<strong>PAM50 subtype</strong>). Then we check whether the explanations agree, make biological sense, survive gene removal, hold up across data splits, and pass sanity tests.</p>
  <div class="tiles" id="tiles"></div>
</div>

<div class="taskbar"><div class="wrap">
  <div class="seg" role="group" aria-label="Prediction task" id="taskseg"></div>
  <span class="hint" id="taskhint"></span>
  <div class="toc"><a href="#q1">Agreement</a><a href="#modules">Modules</a><a href="#q2">Biology</a><a href="#q4">Ablation</a><a href="#q5">Small models</a><a href="#stability">Stability</a><a href="#sanity">Sanity</a></div>
</div></div>

<main class="wrap">
  <section class="q" id="q1">
    <div class="qn">Question 1</div>
    <h2>Do different models identify the same important genes?</h2>
    <p class="answer" id="a-q1"></p>
    <div class="grid2">
      <div class="card"><h3>Top-50 overlap for every model × explainer pair</h3><p class="sub">Jaccard index; 1 = identical top-50 gene lists</p>
        <div class="chart tall" id="c-jaccard"></div><div class="foot"><a data-csv="jaccard">data (CSV)</a></div></div>
      <div class="card"><h3>Agreement relative to chance</h3><p class="sub">Mean Jaccard ÷ the Jaccard expected for two random top-50 lists (dashed line = chance)</p>
        <div class="chart tall" id="c-agree"></div><div class="foot"><a data-csv="pairwise_agreement">data (CSV)</a></div></div>
    </div>
  </section>

  <section class="q" id="modules">
    <div class="qn">Upgrade · Gene modules</div>
    <h2>Do models agree on co-expression <em>modules</em> even when they disagree on genes?</h2>
    <p class="answer" id="a-modules"></p>
    <div class="grid2">
      <div class="card"><h3>Gene-level vs. module-level agreement</h3><p class="sub">Same model × explainer pairs, scored at both levels (× chance)</p>
        <div class="chart" id="c-modagree"></div><div class="foot"><a data-csv="module_vs_gene_agreement">data (CSV)</a></div></div>
      <div class="card"><h3>Removing whole modules vs. the same number of random genes</h3>
        <div class="controls">Model <select id="sel-modabl"></select></div>
        <div class="chart" id="c-modabl"></div><div class="foot"><a data-csv="module_ablation">data (CSV)</a></div></div>
    </div>
    <div class="card" style="margin-top:16px"><h3>Top consensus modules</h3><p class="sub">Modules found by clustering genes on |Pearson r| (training split only), each labelled by its hub gene and best Hallmark match</p>
      <div class="table-scroll" id="t-modules"></div></div>
  </section>

  <section class="q" id="q2">
    <div class="qn">Questions 2 &amp; 3</div>
    <h2>Are the top genes biologically plausible, and do they sit in known cancer pathways?</h2>
    <p class="answer" id="a-q2"></p>
    <div class="grid2">
      <div class="card"><h3>PAM50 genes recovered in each top-50 list</h3><p class="sub">Precision@50: share of the top-50 that are PAM50 genes. Hover to see the hypergeometric p-value.</p>
        <div class="chart tall" id="c-pam50"></div><div class="foot"><a data-csv="known_gene_overlap">data (CSV)</a></div></div>
      <div class="card"><h3>Hallmark pathway enrichment (consensus top-50)</h3><p class="sub">−log10 FDR; dashed line = FDR 0.05</p>
        <div class="chart tall" id="c-enrich"></div><div class="foot"><a data-csv="pathway_enrichment_consensus">data (CSV)</a></div></div>
    </div>
  </section>

  <section class="q" id="q4">
    <div class="qn">Question 4</div>
    <h2>Does removing the top genes change the predictions?</h2>
    <p class="answer" id="a-q4"></p>
    <div class="grid2">
      <div class="card"><h3>Retrain without the top-k genes</h3>
        <div class="controls">Model <select id="sel-abl"></select> Metric <select id="sel-ablm"><option value="roc_auc">ROC-AUC</option><option value="log_loss">log-loss</option></select></div>
        <div class="chart" id="c-abl"></div><div class="foot"><a data-csv="ablation_summary">data (CSV)</a></div></div>
      <div class="card"><h3>Retrain without every PAM50 gene</h3><p class="sub">Balanced accuracy: all genes vs. PAM50 genes removed vs. the same number of random genes removed</p>
        <div class="chart" id="c-pamabl"></div><div class="foot"><a data-csv="ablation_pam50">data (CSV)</a></div></div>
    </div>
  </section>

  <section class="q" id="q5">
    <div class="qn">Question 5</div>
    <h2>Can a small interpretable model perform almost as well as the black boxes?</h2>
    <p class="answer" id="a-q5"></p>
    <div class="card" style="margin-top:18px"><h3>Small models on the top-k consensus genes vs. full models on all 2,000 genes</h3><p class="sub">Balanced accuracy on the held-out test set; dotted lines = full models</p>
      <div class="chart" id="c-sparse"></div><div class="foot"><a data-csv="sparse_models">data (CSV)</a></div></div>
  </section>

  <section class="q" id="stability">
    <div class="qn">Upgrade · Stability across 10 splits</div>
    <h2>Are these findings robust to a different train/test split?</h2>
    <p class="answer" id="a-stab"></p>
    <div class="grid2">
      <div class="card"><h3>Split-to-split noise vs. model-to-model disagreement</h3><p class="sub">Jaccard of top-50 lists: the same method retrained on different splits vs. different models on the same split</p>
        <div class="chart" id="c-stab"></div><div class="foot"><a data-csv="stability/ranking_stability">data (CSV)</a></div></div>
      <div class="card"><h3>Stable core: genes in the consensus top-50 across splits</h3>
        <div class="chart tall" id="c-freq"></div><div class="foot"><a data-csv="stability/gene_selection_frequency">data (CSV)</a></div></div>
    </div>
  </section>

  <section class="q" id="sanity">
    <div class="qn">Upgrade · Sanity checks</div>
    <h2>Do the explanations actually depend on what the model learned?</h2>
    <p class="answer" id="a-sanity"></p>
    <div class="card" style="margin-top:18px"><h3>Label randomization test</h3><p class="sub">Each model is retrained on shuffled labels. Its accuracy should collapse, and its SHAP top-50 should share no more genes with the real model's than chance.</p>
      <div class="table-scroll" id="t-sanity"></div></div>
  </section>

  <section class="q" id="explorer">
    <div class="qn">Gene explorer</div>
    <h2>Every consensus gene, ranked</h2>
    <p class="note">Top 200 genes by consensus score (each gene's rank percentile averaged over every model × explainer combination). Search by gene or pathway; click a column to sort.</p>
    <div class="card"><div class="controls"><input type="search" id="q" placeholder="Search genes or pathways, e.g. ESR1, E2F"></div>
      <div class="table-scroll" id="t-genes"></div></div>
  </section>

  <section class="q" id="figures">
    <div class="qn">All figures</div>
    <h2>Static figures from the pipeline</h2>
    <div class="gallery" id="gallery"></div>
  </section>

  <section class="q" id="methods">
    <div class="qn">Methods</div>
    <h2>How the study works</h2>
    <ul class="clean">
      <li><strong>Data.</strong> TCGA-BRCA RNA-seq (UCSC Xena <code>HiSeqV2</code>, log2(RSEM+1)). Tumor vs. normal uses 1,097 primary tumors and 114 solid-tissue normals. Subtype uses 821 tumors with an RNA-seq PAM50 call (LumA / LumB / Her2 / Basal).</li>
      <li><strong>Leakage control.</strong> Splits are stratified and grouped by patient (64% train / 16% validation / 20% test). The 2,000 most variable genes are selected, and z-scored, on the training split only. Explanations are computed on validation; every reported metric is on the untouched test split.</li>
      <li><strong>Models.</strong> L1 logistic regression, random forest (300 trees), XGBoost (300 trees, depth 4) and an MLP (64-32), all trained with class-balanced sample weights.</li>
      <li><strong>Explainers.</strong> SHAP (Linear / Tree / model-agnostic permutation SHAP for the MLP); permutation importance (Δ log-loss); native importance (|coef|, impurity, gain); partial dependence.</li>
      <li><strong>Agreement.</strong> Jaccard of top-50 lists and Spearman correlation on the union of top lists, reported against an exact chance baseline (hypergeometric simulation).</li>
      <li><strong>Modules.</strong> Average-linkage clustering on 1 − |r| (cut at |r| ≥ 0.6). Module importance is |Σ SHAP| within the module (valid because SHAP is additive) plus grouped permutation, where all genes in a module are shuffled together.</li>
      <li><strong>Biology.</strong> Hypergeometric over-representation (BH-FDR) against MSigDB Hallmark gene sets and the PAM50 genes; the background is the modeled genes.</li>
      <li><strong>Ablation.</strong> Retrain without the top-k genes, whole modules or all PAM50 genes, compared with removing the same number of random genes.</li>
      <li><strong>Stability.</strong> The full train-and-explain loop repeated on 10 different patient-grouped splits.</li>
      <li><strong>Sanity.</strong> Label-randomization test (Adebayo et al., NeurIPS 2018).</li>
    </ul>
    <h3 style="margin-top:28px">Limitations</h3>
    <ul class="clean" id="limits"></ul>
    <h3 style="margin-top:28px">Reproduce</h3>
<pre>git clone __REPO__.git && cd cancer-xai-compare
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m cxai.pipeline --task all   # both tasks
python -m cxai.stability             # 10-split stability study
python -m cxai.site                  # rebuild this page into _site/</pre>
  </section>
</main>
<footer><div class="wrap">Research project, not for clinical use. Data: The Cancer Genome Atlas via UCSC Xena · Hallmark gene sets: MSigDB (CC BY 4.0) · Code: <a href="__REPO__">MIT licence</a>.</div></footer>

<script>
const DATA = __DATA__;
const MODELS = ["logreg", "random_forest", "xgboost", "mlp"];
const PRETTY = {logreg: "Logistic regression", random_forest: "Random forest", xgboost: "XGBoost", mlp: "Neural network"};
const SERIES = {logreg: "--series-1", random_forest: "--series-2", xgboost: "--series-3", mlp: "--series-4"};
const tasks = Object.keys(DATA.tasks);
const PARAMS = new URLSearchParams(location.search);
let task = tasks.includes(PARAMS.get("task")) ? PARAMS.get("task") : (tasks.includes("tumor_normal") ? "tumor_normal" : tasks[0]);
if (["light", "dark"].includes(PARAMS.get("theme"))) document.documentElement.dataset.theme = PARAMS.get("theme");

const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const fmtP = p => (p === undefined || p === null) ? "–" : (Number(p) < 1e-6 ? "&lt; 10<sup>−6</sup>" : "= " + Number(p).toExponential(1));
const pct = x => (100 * x).toFixed(0) + "%";
const T = () => DATA.tasks[task];
const S = () => T().summary;
const SHORT = {logreg: "LogReg", random_forest: "RF", xgboost: "XGB", mlp: "MLP"};
const shortCol = c => { const [m, e] = c.split(":"); return (SHORT[m] || m) + " · " + e; };
const BR = {"same model, different explainer": "same model,<br>different explainer",
            "same explainer, different model": "same explainer,<br>different model",
            "different model and explainer": "different model<br>and explainer"};
const prettyCol = c => { const [m, e] = c.split(":"); return (PRETTY[m] || m) + (e ? " · " + e : ""); };
const esc = s => String(s ?? "").replace(/[&<>"]/g, ch => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[ch]));

function layout(extra = {}) {
  const ink = css("--text-secondary"), grid = css("--grid");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {family: "Inter, system-ui, sans-serif", size: 12, color: ink},
    margin: {l: 52, r: 12, t: 36, b: 48},
    xaxis: {gridcolor: grid, zerolinecolor: grid, linecolor: grid, automargin: true},
    yaxis: {gridcolor: grid, zerolinecolor: grid, linecolor: grid, automargin: true},
    legend: {orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: {size: 11}},
    hoverlabel: {bgcolor: css("--surface-1"), bordercolor: css("--border"), font: {color: css("--text-primary")}},
  }, extra);
}
const CFG = {displayModeBar: false, responsive: true};
const plot = (id, traces, lay) => Plotly.react(id, traces, layout(lay), CFG);

// ---- task switcher & tiles -------------------------------------------------
function renderSeg() {
  const seg = document.getElementById("taskseg");
  seg.innerHTML = tasks.map(t => `<button aria-pressed="${t === task}" data-t="${t}">${DATA.task_names[t]}</button>`).join("");
  seg.querySelectorAll("button").forEach(b => b.onclick = () => {
    task = b.dataset.t; renderAll();
    const u = new URL(location.href); u.searchParams.set("task", task); history.replaceState(null, "", u);
  });
  const ds = S().dataset;
  document.getElementById("taskhint").textContent =
    `${ds.train.n + ds.val.n + ds.test.n} samples · ${ds.classes.join(" / ")} · ${ds.n_genes} genes`;
}

function agreeByType(kind) { return (T().agreement_by_type.find(r => r.comparison === kind) || {}); }

function renderTiles() {
  const s = S(), m = s.test_metrics;
  const bestAuc = Math.max(...Object.values(m).map(r => r.roc_auc));
  const cross = agreeByType("same explainer, different model");
  const sameModel = agreeByType("same model, different explainer");
  const pam = s.known_gene_overlap.consensus || {};
  const sp = T().sparse.filter(r => r.model === "small_logreg");
  const bestBal = Math.max(...Object.values(m).map(r => r.balanced_accuracy));
  const enough = sp.find(r => r.balanced_accuracy >= bestBal - 0.02);
  const tiles = [
    ["Best test ROC-AUC", fmt(bestAuc, 4), `balanced accuracy up to ${fmt(bestBal, 3)}`],
    ["Cross-model gene overlap", fmt(cross.jaccard_top_k, 2), `${fmt(cross.x_chance, 1)}× chance · same model, other explainer: ${fmt(sameModel.jaccard_top_k, 2)}`],
    ["PAM50 genes in consensus top-50", `${pam.overlap ?? "–"} / ${s.top_k}`, `p ${fmtP(pam.p_value)}`],
    ["Genes needed to match black boxes", enough ? String(enough.k) : "> " + Math.max(...sp.map(r => r.k)),
      "logistic regression within 0.02 balanced accuracy of the best full model"],
  ];
  document.getElementById("tiles").innerHTML = tiles.map(([k, v, sub]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`).join("");
}

// ---- narrative (generated from the numbers) -------------------------------
function narrative() {
  const s = S(), n = (DATA.narrative || {})[task] || {};
  const cross = agreeByType("same explainer, different model");
  const same = agreeByType("same model, different explainer");
  document.getElementById("a-q1").innerHTML = (n.q1 ? n.q1 + " " : "") +
    `Different models share <strong>${fmt(cross.jaccard_top_k, 2)}</strong> of their top-50 genes on average (${fmt(cross.x_chance, 1)}× chance), ` +
    `while two explainers applied to the same model share <strong>${fmt(same.jaccard_top_k, 2)}</strong>.`;
  const mv = T().module_vs_gene || [];
  const g = mv.find(r => r.level.startsWith("genes") && r.comparison === "same explainer, different model");
  const mo = mv.find(r => r.level.startsWith("modules") && r.comparison === "same explainer, different model");
  document.getElementById("a-modules").innerHTML = (n.modules ? n.modules + " " : "") + (g && mo ?
    `For different models with the same explainer, agreement is <strong>${fmt(g.x_chance, 1)}×</strong> chance for genes and <strong>${fmt(mo.x_chance, 1)}×</strong> chance for modules ` +
    `(${s.modules.n_modules} modules, ${s.modules.n_multi_gene} with more than one gene).` : "");
  const pam = s.known_gene_overlap.consensus || {};
  const sig = s.significant_pathways_consensus.map(p => p.pathway.replace("HALLMARK_", "").replaceAll("_", " ").toLowerCase());
  document.getElementById("a-q2").innerHTML = (n.q2 ? n.q2 + " " : "") +
    `The consensus top-50 contains <strong>${pam.overlap}</strong> PAM50 genes (p ${fmtP(pam.p_value)}). ` +
    (sig.length ? `Significant Hallmark pathways (FDR &lt; 0.05): ${sig.join(", ")}.` : "No Hallmark pathway reaches FDR &lt; 0.05.");
  document.getElementById("a-q4").innerHTML = n.q4 || "";
  document.getElementById("a-q5").innerHTML = n.q5 || "";
  document.getElementById("a-sanity").innerHTML = n.sanity || "";
  const st = DATA.stability && DATA.stability.summary[task];
  document.getElementById("a-stab").innerHTML = st ? (n.stability ? n.stability + " " : "") +
    `Across ${st.n_splits} splits, the same method's top-50 overlaps itself by <strong>${fmt(st.within_method_stability_mean, 2)}</strong> (Jaccard), ` +
    `while different models overlap by <strong>${fmt(st.cross_model_agreement_mean, 2)}</strong>. ` +
    `${st.stable_core_genes_ge_80pct.length} genes are in the consensus top-50 in at least 80% of splits.` : "Stability results not available.";
  document.getElementById("limits").innerHTML = ((DATA.narrative || {}).limitations || []).map(l => `<li>${l}</li>`).join("");
}

// ---- charts ----------------------------------------------------------------
function seqScale() { return [[0, css("--seq-0")], [0.15, css("--seq-1")], [0.35, css("--seq-2")], [0.6, css("--seq-3")], [0.85, css("--seq-4")], [1, css("--seq-5")]]; }

function chartJaccard() {
  const j = T().jaccard, labels = j.labels.map(shortCol);
  plot("c-jaccard", [{type: "heatmap", z: j.z, x: labels, y: labels, colorscale: seqScale(), zmin: 0, zmax: 1, xgap: 2, ygap: 2,
    hovertemplate: "%{y}<br>%{x}<br>Jaccard %{z:.2f}<extra></extra>", colorbar: {thickness: 10, outlinewidth: 0}}],
    {margin: {l: 120, r: 10, t: 8, b: 110}, xaxis: {tickangle: -45, tickfont: {size: 10}, showgrid: false},
     yaxis: {autorange: "reversed", tickfont: {size: 10}, showgrid: false}});
}

function chartAgree() {
  const rows = T().agreement_by_type;
  const order = ["same model, different explainer", "same explainer, different model", "different model and explainer"];
  const r = order.map(o => rows.find(x => x.comparison === o) || {});
  plot("c-agree", [{type: "bar", x: order.map(o => BR[o]), y: r.map(x => x.x_chance), marker: {color: css("--series-1"), cornerradius: 4},
    text: r.map(x => fmt(x.x_chance, 1) + "×"), textposition: "outside", cliponaxis: false,
    customdata: r.map(x => x.jaccard_top_k), hovertemplate: "%{y:.1f}× chance<br>mean Jaccard %{customdata:.3f}<extra></extra>"}],
    {yaxis: {title: "× chance", rangemode: "tozero"}, shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: {dash: "dash", color: css("--ref"), width: 1.5}}],
     showlegend: false, bargap: 0.45});
}

function chartModAgree() {
  const mv = T().module_vs_gene || [];
  if (!mv.length) return Plotly.purge("c-modagree");
  const levels = [...new Set(mv.map(r => r.level))];
  const order = ["same model, different explainer", "same explainer, different model", "different model and explainer"];
  const traces = levels.map((lv, i) => ({type: "bar", name: lv, x: order.map(o => BR[o]),
    y: order.map(o => (mv.find(r => r.level === lv && r.comparison === o) || {}).x_chance),
    marker: {color: css(i ? "--series-2" : "--series-1"), cornerradius: 4},
    hovertemplate: lv + "<br>%{y:.1f}× chance<extra></extra>"}));
  plot("c-modagree", traces, {barmode: "group", bargap: 0.3, bargroupgap: 0.08, yaxis: {title: "× chance", rangemode: "tozero"},
    shapes: [{type: "line", xref: "paper", x0: 0, x1: 1, y0: 1, y1: 1, line: {dash: "dash", color: css("--ref"), width: 1.5}}]});
}

function fillModelSelect(id, onchange) {
  const sel = document.getElementById(id);
  const prev = sel.value || "xgboost";
  sel.innerHTML = MODELS.map(m => `<option value="${m}">${PRETTY[m]}</option>`).join("");
  sel.value = MODELS.includes(prev) ? prev : "xgboost";
  sel.onchange = onchange;
}

function chartModAbl() {
  const m = document.getElementById("sel-modabl").value;
  const d = (T().module_ablation || []).filter(r => r.model === m);
  if (!d.length) return Plotly.purge("c-modabl");
  const line = (rem, name, color, dash) => {
    const s = d.filter(r => r.removed === rem).sort((a, b) => a.genes_removed - b.genes_removed);
    return {type: "scatter", mode: "lines+markers", name, x: s.map(r => r.genes_removed), y: s.map(r => r.balanced_accuracy),
      text: s.map(r => r.modules), line: {color, width: 2, dash}, marker: {size: 8},
      hovertemplate: name + "<br>%{x} genes removed (%{text} modules)<br>balanced acc %{y:.3f}<extra></extra>"};
  };
  plot("c-modabl", [line("top_modules", "top consensus modules", css("--series-1"), "solid"),
                    line("random", "same # random genes", css("--ref"), "dash")],
       {xaxis: {title: "genes removed"}, yaxis: {title: "balanced accuracy"}});
}

function chartPam50() {
  const k = T().known.slice().sort((a, b) => a.precision_at_k - b.precision_at_k);
  plot("c-pam50", [{type: "bar", orientation: "h", y: k.map(r => r.ranking === "consensus" ? "Consensus" : prettyCol(r.ranking)),
    x: k.map(r => r.precision_at_k), marker: {color: k.map(r => r.ranking === "consensus" ? css("--series-2") : css("--series-1")), cornerradius: 4},
    customdata: k.map(r => [r.overlap, r.p_value]),
    hovertemplate: "%{y}<br>%{customdata[0]} PAM50 genes in top-50<br>p = %{customdata[1]:.2e}<extra></extra>"}],
    {margin: {l: 170, r: 16, t: 8, b: 40}, xaxis: {title: "precision@50", tickformat: ".0%", rangemode: "tozero"}, showlegend: false, bargap: 0.35});
}

function chartEnrich() {
  const e = T().enrichment.slice().reverse();
  plot("c-enrich", [{type: "bar", orientation: "h", y: e.map(r => r.pathway), x: e.map(r => -Math.log10(Math.max(r.fdr, 1e-300))),
    marker: {color: css("--series-1"), cornerradius: 4}, customdata: e.map(r => [r.overlap, r.list_size, r.fold_enrichment, r.fdr]),
    hovertemplate: "%{y}<br>%{customdata[0]}/%{customdata[1]} genes · %{customdata[2]:.1f}× enriched<br>FDR %{customdata[3]:.2e}<extra></extra>"}],
    {margin: {l: 210, r: 16, t: 8, b: 40}, xaxis: {title: "−log10 FDR", rangemode: "tozero"}, showlegend: false, bargap: 0.35,
     shapes: [{type: "line", yref: "paper", y0: 0, y1: 1, x0: -Math.log10(0.05), x1: -Math.log10(0.05), line: {dash: "dash", color: css("--ref"), width: 1.5}}]});
}

function chartAbl() {
  const m = document.getElementById("sel-abl").value, metric = document.getElementById("sel-ablm").value;
  const d = T().ablation.filter(r => r.model === m);
  const base = (d.find(r => r.removed === "none") || {})[metric];
  const line = (rem, name, color, dash) => {
    const s = d.filter(r => r.removed === rem).sort((a, b) => a.k - b.k);
    return {type: "scatter", mode: "lines+markers", name, x: s.map(r => r.k), y: s.map(r => r[metric]), line: {color, width: 2, dash}, marker: {size: 8},
      error_y: rem === "random" ? {type: "data", array: s.map(r => r[metric + "_sd"] || 0), color, thickness: 1, width: 3} : undefined,
      hovertemplate: name + "<br>%{x} genes removed<br>%{y:.4f}<extra></extra>"};
  };
  plot("c-abl", [line("own_top", "model's own top genes", css("--series-1"), "solid"),
                 line("consensus_top", "consensus top genes", css("--series-2"), "solid"),
                 line("random", "random genes (mean ± sd)", css("--ref"), "dash")],
    {xaxis: {title: "genes removed (log scale)", type: "log"}, yaxis: {title: metric === "roc_auc" ? "ROC-AUC" : "log-loss"},
     shapes: base === undefined ? [] : [{type: "line", xref: "paper", x0: 0, x1: 1, y0: base, y1: base, line: {dash: "dot", color: css("--ref"), width: 1}}]});
}

function chartPamAbl() {
  const rows = T().pam50_removal, metrics = S().test_metrics;
  const val = (m, rem) => (rows.find(r => r.model === m && r.removed === rem) || {}).balanced_accuracy;
  const mk = (name, ys, color) => ({type: "bar", name, x: MODELS.map(m => PRETTY[m]), y: ys, marker: {color, cornerradius: 4},
    hovertemplate: name + "<br>%{x}: %{y:.3f}<extra></extra>"});
  plot("c-pamabl", [mk("all genes", MODELS.map(m => metrics[m].balanced_accuracy), css("--series-1")),
                    mk("without PAM50 genes", MODELS.map(m => val(m, "pam50")), css("--series-2")),
                    mk("without random genes", MODELS.map(m => val(m, "random")), css("--ref"))],
    {barmode: "group", bargap: 0.3, bargroupgap: 0.08, yaxis: {title: "balanced accuracy"}});
}

function chartSparse() {
  const sp = T().sparse, metrics = S().test_metrics;
  const names = [...new Set(sp.map(r => r.model))];
  const traces = names.map((n, i) => {
    const s = sp.filter(r => r.model === n).sort((a, b) => a.k - b.k);
    const label = n === "small_logreg" ? "small logistic regression" : n.replace("tree_depth", "decision tree, depth ");
    return {type: "scatter", mode: "lines+markers", name: label, x: s.map(r => r.k), y: s.map(r => r.balanced_accuracy),
      line: {color: css(i ? "--series-2" : "--series-1"), width: 2}, marker: {size: 8},
      hovertemplate: label + "<br>%{x} genes<br>balanced acc %{y:.3f}<extra></extra>"};
  });
  const shapes = [], ann = [];
  MODELS.forEach(m => { const y = metrics[m].balanced_accuracy;
    shapes.push({type: "line", xref: "paper", x0: 0, x1: 1, y0: y, y1: y, line: {dash: "dot", color: css(SERIES[m]), width: 1.5}});
    traces.push({type: "scatter", mode: "lines", name: PRETTY[m] + " (all genes)", x: [null], y: [null], line: {dash: "dot", color: css(SERIES[m])}});
  });
  plot("c-sparse", traces, {xaxis: {title: "genes in the small model (log scale)", type: "log", tickvals: sp.map(r => r.k)},
    yaxis: {title: "balanced accuracy"}, shapes, annotations: ann});
}

function chartStability() {
  const st = DATA.stability;
  if (!st || !st.summary[task]) { Plotly.purge("c-stab"); Plotly.purge("c-freq"); return; }
  const within = st.within.filter(r => r.task === task && r.ranking !== "consensus");
  const cross = st.cross_pairs.filter(r => r.task === task);
  const box = (name, ys, labels, color) => ({type: "box", name, y: ys, text: labels, boxpoints: "all", jitter: 0.4, pointpos: 0,
    marker: {color, size: 7}, line: {color, width: 1.5}, fillcolor: "rgba(0,0,0,0)", hovertemplate: "%{text}<br>Jaccard %{y:.2f}<extra></extra>"});
  plot("c-stab", [box("same method,<br>different split", within.map(r => r.mean_jaccard_across_splits), within.map(r => prettyCol(r.ranking)), css("--series-1")),
                  box("different model,<br>same split", cross.map(r => r.jaccard), cross.map(r => prettyCol(r.a) + " vs " + prettyCol(r.b)), css("--series-2"))],
    {yaxis: {title: "Jaccard of top-50", range: [0, 1]}, showlegend: false});
  const f = st.freq.filter(r => r.task === task).slice().reverse();
  plot("c-freq", [{type: "bar", orientation: "h", y: f.map(r => r.gene), x: f.map(r => r.fraction_of_splits),
    marker: {color: f.map(r => r.fraction_of_splits >= 0.8 ? css("--series-1") : css("--seq-1")), cornerradius: 3},
    hovertemplate: "%{y}: in %{x:.0%} of splits<extra></extra>"}],
    {margin: {l: 90, r: 16, t: 8, b: 40}, xaxis: {title: "share of splits", tickformat: ".0%", range: [0, 1.02]}, showlegend: false, bargap: 0.25,
     shapes: [{type: "line", yref: "paper", y0: 0, y1: 1, x0: 0.8, x1: 0.8, line: {dash: "dash", color: css("--ref"), width: 1.5}}]});
}

// ---- tables ----------------------------------------------------------------
function table(el, rows, cols, opts = {}) {
  let sortKey = opts.sort, asc = false;
  const draw = () => {
    const q = (opts.filter ? opts.filter() : "").toLowerCase();
    let r = rows.filter(row => !q || cols.some(c => String(row[c.key] ?? "").toLowerCase().includes(q)));
    if (sortKey) r = r.slice().sort((a, b) => { const x = a[sortKey], y = b[sortKey];
      const c = (typeof x === "number" && typeof y === "number") ? x - y : String(x).localeCompare(String(y)); return asc ? c : -c; });
    el.innerHTML = `<table><thead><tr>${cols.map(c => `<th data-k="${c.key}">${c.label}${sortKey === c.key ? (asc ? " ▲" : " ▼") : ""}</th>`).join("")}</tr></thead><tbody>` +
      r.map(row => `<tr>${cols.map(c => `<td class="${c.wrap ? "wrap-cell" : ""}">${c.fmt ? c.fmt(row[c.key], row) : esc(row[c.key])}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
    el.querySelectorAll("th").forEach(th => th.onclick = () => { asc = sortKey === th.dataset.k ? !asc : false; sortKey = th.dataset.k; draw(); });
  };
  draw();
  return draw;
}

function geneTable() {
  const rows = T().consensus.map((r, i) => Object.assign({rank: i + 1}, r));
  const binary = "log2FC_tumor_vs_normal" in (rows[0] || {});
  const cols = [{key: "rank", label: "#"}, {key: "gene", label: "Gene", fmt: v => `<strong>${esc(v)}</strong>`},
    {key: "consensus_score", label: "Consensus", fmt: v => fmt(v, 3)}, {key: "n_methods_top50", label: "In top-50 of"}];
  if (binary) cols.push({key: "log2FC_tumor_vs_normal", label: "log2FC tumor/normal", fmt: v => (v > 0 ? "+" : "") + fmt(v, 2)});
  else cols.push({key: "highest_in", label: "Highest in"}, {key: "lowest_in", label: "Lowest in"});
  cols.push({key: "PAM50", label: "PAM50", fmt: v => v ? '<span class="pill">PAM50</span>' : ""},
            {key: "hallmark_pathways", label: "Hallmark pathways", wrap: true, fmt: v => esc(String(v || "").replaceAll(";", ", ").replaceAll("_", " ").toLowerCase())});
  const input = document.getElementById("q");
  const draw = table(document.getElementById("t-genes"), rows, cols, {filter: () => input.value});
  input.oninput = draw;
}

function moduleTable() {
  const rows = T().modules || [];
  table(document.getElementById("t-modules"), rows, [
    {key: "label", label: "Module · hub gene", fmt: v => `<strong>${esc(v)}</strong>`}, {key: "size", label: "Genes"},
    {key: "consensus_score", label: "Consensus", fmt: v => fmt(v, 3)},
    {key: "hallmark", label: "Best Hallmark match", fmt: v => esc(String(v || "–").replaceAll("_", " ").toLowerCase())},
    {key: "genes", label: "Members", wrap: true, fmt: v => { const g = String(v || "").split(","); return esc(g.slice(0, 14).join(", ") + (g.length > 14 ? ` … (+${g.length - 14})` : "")); }}]);
}

function sanityTable() {
  table(document.getElementById("t-sanity"), T().sanity, [
    {key: "model", label: "Model", fmt: v => PRETTY[v] || v},
    {key: "test_roc_auc_random_labels", label: "Test ROC-AUC (shuffled labels)", fmt: v => fmt(v, 3)},
    {key: "real", label: "Test ROC-AUC (real labels)", fmt: (_, r) => fmt(S().test_metrics[r.model].roc_auc, 3)},
    {key: "jaccard_vs_real_model", label: "SHAP top-50 overlap with real model", fmt: v => fmt(v, 3)},
    {key: "chance", label: "Chance overlap", fmt: () => fmt(S().chance_jaccard_genes.mean, 3)}]);
}

function gallery() {
  const items = T().figures.map(f => [`figures/${task}/${f}`, f]);
  if (DATA.stability) DATA.stability.figures.forEach(f => items.push([`figures/stability/${f}`, "stability · " + f]));
  document.getElementById("gallery").innerHTML = items.map(([src, name]) =>
    `<a href="${src}" target="_blank" rel="noopener"><img loading="lazy" src="${src}" alt="${esc(name)}"><span>${esc(name.replace(".png", "").replaceAll("_", " "))}</span></a>`).join("");
}

function csvLinks() {
  document.querySelectorAll("[data-csv]").forEach(a => {
    const k = a.dataset.csv;
    const name = k === "jaccard" ? `jaccard_top${S().top_k}` : k;
    a.href = k.startsWith("stability/") ? `data/${k}.csv` : `data/${task}/${name}.csv`;
  });
}

function renderCharts() {
  chartJaccard(); chartAgree(); chartModAgree(); chartModAbl(); chartPam50(); chartEnrich();
  chartAbl(); chartPamAbl(); chartSparse(); chartStability();
}

function renderAll() {
  renderSeg(); renderTiles(); narrative();
  fillModelSelect("sel-abl", chartAbl); fillModelSelect("sel-modabl", chartModAbl);
  document.getElementById("sel-ablm").onchange = chartAbl;
  renderCharts(); geneTable(); moduleTable(); sanityTable(); gallery(); csvLinks();
}

// ---- theme ------------------------------------------------------------------
(function theme() {
  const KEY = "cxai-theme";
  let saved = null; try { saved = localStorage.getItem(KEY); } catch (e) {}
  if (saved && !PARAMS.get("theme")) document.documentElement.dataset.theme = saved;
  document.getElementById("theme").onclick = () => {
    const dark = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    const next = dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem(KEY, next); } catch (e) {}
    renderCharts();
  };
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", renderCharts);
})();

if (tasks.length) renderAll();
else document.querySelector("main").innerHTML = '<p class="empty">No results found. Run the pipeline first.</p>';
</script>
</body>
</html>
"""
