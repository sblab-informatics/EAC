#!/usr/bin/env python3
"""
Figure 3 — Panel C: GO/pathway enrichment of promoters that GAINED a G4.

Query (per column) = promoters G4- at NG that became G4+ at the target stage
(gained-G4), from the corrected reproducible BG4 sets:
  col 1  NG -> ES      (G4- NG, G4+ ES_EAC)
  col 2  NG -> LS      (G4- NG, G4+ LS_EAC)
  col 3  NG -> either  (G4- NG, G4+ ES_EAC OR LS_EAC)

TWO backgrounds (the choice is the whole point — stated on every figure):
  BG1 "all_genes"  : vs ALL genes (38,841). PERMISSIVE; G4+ promoters are largely
                     expressed, so BG1 enrichment is partly an EXPRESSION confound.
                     Query = FULL gained set.
  BG2 "expressed"  : vs ALL expressed genes (TPM>0.1 in any stage; n=17,863).
                     Query = EXPRESSED gainers (a SUBSET of the background, as
                     Fisher over-representation requires). Expression-matched —
                     isolates G4-gain enrichment BEYOND being expressed.
                     (NOT "gainers vs non-gainers": that would put the query
                     outside the background and break the test.)

FIVE ontologies: GO BP 2025, KEGG 2026, Reactome 2024, MSigDB Hallmark 2020,
MSigDB Oncogenic. => 5 ontologies x 2 backgrounds = 10 heatmaps (3 cols each).

Heatmap: rows = union of top-N FDR-significant terms per column; colour =
signed -log10(FDR q)  (red = enriched, blue = depleted, white = NS) in RdBu_r,
capped at +/-50. Effect size (odds ratio) + full stats saved per heatmap.
"""
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA_DIR = SCRIPT_DIR / 'data'
PNG_DIR = SCRIPT_DIR / 'figures' / 'png'
SVG_DIR = SCRIPT_DIR / 'figures' / 'svg'
GMT_DIR = SCRIPT_DIR / 'reference_data' / 'gene_sets'
RNA_DIR = SCRIPT_DIR / 'data' / 'rnaseq'
for d in (PNG_DIR, SVG_DIR):
    d.mkdir(parents=True, exist_ok=True)
config.setup_style()

MIN_P = 1e-300
LOG10_MIN_P = math.log10(MIN_P)
TOP_N = 15
FDR_THR = 0.05
CAP = 50.0   # colour cap on signed -log10 q

ONTOLOGIES = {
    'GO_BP': 'go_biological_process_2025.gmt',
    'KEGG': 'kegg_2026.gmt',
    'Reactome': 'reactome_pathways_2024.gmt',
    'Hallmark': 'msigdb_hallmark_2020.gmt',
    'Oncogenic': 'msigdb_oncogenic_signatures.gmt',
}
COLS = ['NG_to_ES', 'NG_to_LS', 'NG_to_either']
COL_LABEL = {'NG_to_ES': 'NG→ES', 'NG_to_LS': 'NG→LS', 'NG_to_either': 'NG→ES/LS'}


def bh_fdr(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / (np.arange(1, n + 1)))[::-1])[::-1]
    return np.clip(q, 0, 1)


def enrich(query, background, gene_sets):
    """Fisher exact over-representation of `query` in each gene set, within
    `background`. Returns DataFrame with odds ratio, p, BH-q (and log10 forms)."""
    bg = set(background)
    q = set(query) & bg
    N = len(bg)
    nq = len(q)
    rows = []
    for term, gs in gene_sets.items():
        gsb = gs & bg
        m = len(gsb)
        if m < 5:
            continue
        k = len(q & gsb)                       # query in term
        if k == 0:
            continue
        # 2x2: [[k, nq-k],[m-k, N-nq-(m-k)]]
        table = [[k, nq - k], [m - k, N - nq - (m - k)]]
        orr, p = fisher_exact(table, alternative='two-sided')
        # direction: enriched if observed k > expected
        expected = nq * m / N if N else 0
        direction = 1.0 if k >= expected else -1.0
        rows.append({'term': term, 'overlap': k, 'term_size_in_bg': m,
                     'query_size_in_bg': nq, 'background_size': N,
                     'gene_ratio': k / nq if nq else 0, 'odds_ratio': orr,
                     'direction': direction, 'p_raw': p,
                     'genes': ';'.join(sorted(q & gsb))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df['q_fdr'] = bh_fdr(df['p_raw'].values)
    df['p_raw'] = df['p_raw'].clip(lower=MIN_P)
    df['q_fdr'] = df['q_fdr'].clip(lower=MIN_P)
    df['log10_p_raw'] = np.log10(df['p_raw'])
    df['log10_q_fdr'] = np.log10(df['q_fdr'])
    # signed -log10 q (red=enriched, blue=depleted)
    df['signed_neglog10_q'] = df['direction'] * (-df['log10_q_fdr'])
    return df


def make_heatmap(per_col, ontology, bg_key, bg_desc, query_desc):
    """per_col: {col: enrichment_df}. Build a 3-col signed -log10 q heatmap."""
    # rows = union of top-N FDR-sig terms per column, then RANKED by the first
    # column (NG->ES) effect size (log2 odds ratio), descending.
    rowset = []
    for col in COLS:
        d = per_col[col]
        if d.empty:
            continue
        sig = d[(d['q_fdr'] < FDR_THR) & (d['direction'] > 0)].nsmallest(TOP_N, 'q_fdr')
        rowset += sig['term'].tolist()
    rows = list(dict.fromkeys(rowset))
    if not rows:
        print(f'    [no sig terms] {ontology} / {bg_key}')
        return
    first = per_col[COLS[0]].set_index('term')   # NG->ES
    def _rank(term):
        orr = first['odds_ratio'].get(term, np.nan) if term in first.index else np.nan
        return np.log2(orr) if (orr and np.isfinite(orr) and orr > 0) else -np.inf
    rows = sorted(rows, key=_rank, reverse=True)
    # COLOUR = log2(odds ratio) [effect size]; STARS = FDR q. Scaled to the data.
    Meff = np.full((len(rows), len(COLS)), np.nan)
    Mq = np.full((len(rows), len(COLS)), np.nan)
    for ci, col in enumerate(COLS):
        d = per_col[col].set_index('term')
        for ri, term in enumerate(rows):
            if term in d.index:
                orr = d['odds_ratio'].get(term, np.nan)
                Meff[ri, ci] = np.log2(orr) if (orr and np.isfinite(orr) and orr > 0) else 0.0
                Mq[ri, ci] = d['q_fdr'].get(term, np.nan)
    Meff = np.nan_to_num(Meff, nan=0.0)

    h = max(3.0, 0.32 * len(rows) + 1.6)
    fig, ax = plt.subplots(figsize=(config.COL_FULL * 0.85, h))
    im = config.effect_heatmap(ax, Meff, [t[:48] for t in rows],
                               [COL_LABEL[c] for c in COLS], star_q=Mq)
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label('log2(odds ratio)\n(red=enriched, blue=depleted)', fontsize=11)
    ax.set_title('Gained-G4 promoter enrichment · {}\nbackground: {}\n* q<0.05  ** q<0.01  *** q<0.001  **** q<1e-10'
                 .format(ontology, bg_desc), fontsize=12, fontweight='bold')
    config.panel_label(ax, 'C')
    fig.tight_layout()
    fname = f'fig3C_GO_gained_{ontology}_{bg_key}'
    fig.savefig(PNG_DIR / f'{fname}.png', dpi=300, bbox_inches='tight', facecolor='white')
    svg = SVG_DIR / f'{fname}.svg'
    fig.savefig(svg, format='svg', bbox_inches='tight', facecolor='white')
    with open(svg) as fh:
        t = fh.read()
    with open(svg, 'w') as fh:
        fh.write(config.fix_svg_for_affinity(t))
    plt.close(fig)
    # RAW DATA: full enrichment per column (long form)
    raw = []
    for col in COLS:
        d = per_col[col].copy()
        d.insert(0, 'transition', col)
        d.insert(0, 'ontology', ontology)
        d.insert(0, 'background', bg_key)
        raw.append(d)
    pd.concat(raw, ignore_index=True).to_csv(DATA_DIR / f'{fname}_rawdata.csv', index=False)
    print(f'    {fname}: {len(rows)} rows')


# ---- query + background gene sets -------------------------------------------
g4 = pd.read_csv(DATA_DIR / 'promoter_g4_status_corrected.csv')
tpm = pd.read_csv(RNA_DIR / 'gene_tpm_matrix.csv')
meta = pd.read_csv(RNA_DIR / 'sample_metadata_corrected.csv')
gcol = 'gene_name' if 'gene_name' in tpm.columns else tpm.columns[0]
sm = {}
for st in ['NG', 'ES_EAC', 'LS_EAC']:
    sids = [s for s in meta[meta['analytical_condition'] == st]['sample_id'] if s in tpm.columns]
    sm[st] = tpm.set_index(gcol)[sids].mean(axis=1)
expr_any = set(sm['NG'][(sm['NG'] > 0.1) | (sm['ES_EAC'] > 0.1) | (sm['LS_EAC'] > 0.1)].index)

gained = {
    'NG_to_ES': set(g4[(g4.g4_NG == 'G4-') & (g4.g4_ES_EAC == 'G4+')]['gene']),
    'NG_to_LS': set(g4[(g4.g4_NG == 'G4-') & (g4.g4_LS_EAC == 'G4+')]['gene']),
    'NG_to_either': set(g4[(g4.g4_NG == 'G4-') & ((g4.g4_ES_EAC == 'G4+') | (g4.g4_LS_EAC == 'G4+'))]['gene']),
}
all_genes = set(g4['gene'])
# BG2 = ALL EXPRESSED genes (TPM>0.1 any). The query (expressed gainers) is a
# SUBSET of this background — Fisher over-representation requires query ⊆ bg.
# This is the expression-matched test: among expressed genes, are gainers
# over-represented in term X? (NOT "gainers vs non-gainers", which would put the
# query outside the background and break the test.)
bg_expressed = expr_any
print('query sizes (full gained):', {k: len(v) for k, v in gained.items()})
print('all genes:', len(all_genes), '| expressed bg:', len(bg_expressed))

# Four versions. This is CHROMATIN ARCHITECTURE — where G4 secondary structures
# form across the genome — so the PRIMARY, biologically meaningful question is
# "what functions do G4-GAINING promoters mark, vs the whole genome": that is
# V1 (gained vs all genes). Expression is a downstream consequence, NOT the
# relevant axis here, so the expression-matched versions (V3/V4) are kept only
# as a methodological robustness footnote against the "it's just expression"
# objection — NOT the headline.
#   restrict_expr=True  -> query limited to expressed (TPM>0.1) gainers
VERSIONS = {
    'V1_gained_vs_allgenes':      (all_genes,   False, '[PRIMARY] gained-G4 promoters vs ALL genes (n={:,}) — where G4-gain lands in the genome'.format(len(all_genes))),
    'V2_gainedExpr_vs_allgenes':  (all_genes,   True,  'expressed gained-G4 vs ALL genes (n={:,})'.format(len(all_genes))),
    'V3_gained_vs_expressed':     (bg_expressed, False, 'gained-G4 vs expressed-gene bg (n={:,}) — expression robustness'.format(len(bg_expressed))),
    'V4_gainedExpr_vs_expressed': (bg_expressed, True,  '[control] expressed gained-G4 vs expressed bg (n={:,}) — expression-matched footnote'.format(len(bg_expressed))),
}

for ont_key, gmt in ONTOLOGIES.items():
    gene_sets = config.load_gmt(str(GMT_DIR / gmt))
    print(f'== {ont_key} ({len(gene_sets)} terms) ==')
    for ver_key, (bg, restrict_expr, desc) in VERSIONS.items():
        per_col = {}
        for col in COLS:
            q = (gained[col] & expr_any) if restrict_expr else gained[col]
            per_col[col] = enrich(q, bg, gene_sets)
        make_heatmap(per_col, ont_key, ver_key, desc, 'restrict_expr' if restrict_expr else 'full')
print('done — GO gained enrichment heatmaps (5 ontologies x 4 versions)')
