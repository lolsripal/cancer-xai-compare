"""Biological plausibility checks: known-gene overlap and pathway enrichment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, hypergeom

RESOURCES = Path(__file__).resolve().parents[2] / "resources"
HALLMARK_GMT = RESOURCES / "h.all.v2024.1.Hs.symbols.gmt"
KNOWN_GENES = RESOURCES / "breast_cancer_genes.tsv"


def read_gmt(path: str | Path = HALLMARK_GMT) -> dict[str, set[str]]:
    sets = {}
    for line in Path(path).read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) > 2:
            sets[parts[0]] = set(parts[2:])
    return sets


def read_known_genes(path: str | Path = KNOWN_GENES) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#")


def overlap_test(hits: list[str], reference: set[str], background: list[str]) -> dict:
    """One-sided hypergeometric test for over-representation of ``reference`` in ``hits``."""
    bg = set(background)
    ref = reference & bg
    hit = set(hits) & bg
    k = len(hit & ref)
    M, n, N = len(bg), len(ref), len(hit)
    p = hypergeom.sf(k - 1, M, n, N) if k > 0 else 1.0
    expected = N * n / M if M else np.nan
    return {"overlap": k, "list_size": N, "set_size_in_background": n,
            "expected": expected, "fold_enrichment": k / expected if expected else np.nan,
            "p_value": float(p), "genes": ",".join(sorted(hit & ref))}


def enrichment(
    hits: list[str],
    background: list[str],
    gene_sets: dict[str, set[str]],
    min_size: int = 5,
) -> pd.DataFrame:
    rows = []
    for name, members in gene_sets.items():
        if len(members & set(background)) < min_size:
            continue
        rows.append({"pathway": name, **overlap_test(hits, members, background)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["fdr"] = false_discovery_control(df["p_value"].values, method="bh")
    return df.sort_values("p_value").reset_index(drop=True)


def gene_to_pathways(gene_sets: dict[str, set[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for name, members in gene_sets.items():
        for g in members:
            out.setdefault(g, []).append(name.replace("HALLMARK_", ""))
    return out
