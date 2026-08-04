#!/usr/bin/env python3
"""
Figure 3 — Panel M: Venn of G4-GAINED promoters (ES vs LS) + GO of the unique sets.

Gained-G4 promoter = G4- at NG -> G4+ at the stage (acquisition; same definition
as the gained-G4 GO panels). We split the gained promoters by ES vs LS:

  ES n LS  (both)  — gained early, retained        : ~3,873
  ES-only          — gained at ES, LOST by LS (TRANSIENT/early-only) : ~905
  LS-only          — gained only at LS (LATE-acquired)               : ~1,753

M1  proportional Venn of gained-G4(ES) vs gained-G4(LS).
M2  GO/pathway enrichment of the ES-ONLY (transient) and LS-ONLY (late) unique
    promoter sets, each vs ALL genes background — what programmes does each
    timing class mark? 5 ontologies, effect-size (log2 OR) heatmap + FDR stars.

Effect-size led; p bounded (1e-300 floor + log10); unit = promoters (NOT donors).
Each panel writes its own *_rawdata.csv + a master GO table.
"""
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA = SCRIPT_DIR / 'data'
PNG = SCRIPT_DIR / 'figures' / 'png'
SVG = SCRIPT_DIR / 'figures' / 'svg'
GMT_DIR = SCRIPT_DIR / 'reference_data' / 'gene_sets'
config.setup_style()
MIN_P = 1e-300
TOP_N = 15
FDR_THR = 0.05
LOG2_CAP = 6.0
ONTOLOGIES = {'GO_BP': 'go_biological_process_2025.gmt', 'KEGG': 'kegg_2026.gmt',
              'Reactome': 'reactome_pathways_2024.gmt', 'Hallmark': 'msigdb_hallmark_2020.gmt',
              'Oncogenic': 'msigdb_oncogenic_signatures.gmt'}
ES_COL, LS_COL = '#F4A582', '#B2182B'   # ES / LS condition colours


# ---- gained-G4 sets --------------------------------------------------------
g = pd.read_csv(DATA / 'promoter_g4_status_corrected.csv')
gained_ES = set(g[(g.g4_NG == 'G4-') & (g.g4_ES_EAC == 'G4+')].gene)
gained_LS = set(g[(g.g4_NG == 'G4-') & (g.g4_LS_EAC == 'G4+')].gene)
both = gained_ES & gained_LS
es_only = gained_ES - gained_LS
ls_only = gained_LS - gained_ES
all_genes = set(g.gene)
pd.DataFrame([
    {'set': 'ES_gained', 'n': len(gained_ES)}, {'set': 'LS_gained', 'n': len(gained_LS)},
    {'set': 'ES_and_LS', 'n': len(both)}, {'set': 'ES_only_transient', 'n': len(es_only)},
    {'set': 'LS_only_late', 'n': len(ls_only)},
]).to_csv(DATA / 'fig3M_gained_ES_LS_venn_rawdata.csv', index=False)
pd.DataFrame({'gene': sorted(es_only)}).to_csv(DATA / 'fig3M_ES_only_genes.csv', index=False)
pd.DataFrame({'gene': sorted(ls_only)}).to_csv(DATA / 'fig3M_LS_only_genes.csv', index=False)


# ---- M1: Venn (clean: fixed equal radii, controlled overlap, no label clash) -
def draw_venn(ax, only_a, only_b, both_n, label_a, label_b, col_a, col_b):
    R = 1.45            # equal radii (clean, readable; areas not proportional)
    cx = 0.95           # half the horizontal centre separation (overlap ~ moderate)
    ca, cb = (-cx, 0), (cx, 0)
    ax.add_patch(Circle(ca, R, fc=col_a, ec=col_a, lw=2.0, alpha=0.40, zorder=2))
    ax.add_patch(Circle(cb, R, fc=col_b, ec=col_b, lw=2.0, alpha=0.40, zorder=2))
    # numbers: left-lobe centre, lens centre (x=0), right-lobe centre — well separated
    ax.text(-cx - 0.55, 0, '{:,}'.format(only_a), ha='center', va='center',
            fontsize=17, fontweight='bold', color=col_a, zorder=4)
    ax.text(cx + 0.55, 0, '{:,}'.format(only_b), ha='center', va='center',
            fontsize=17, fontweight='bold', color=col_b, zorder=4)
    ax.text(0, 0, '{:,}'.format(both_n), ha='center', va='center',
            fontsize=17, fontweight='bold', color='#222', zorder=4)
    # set labels ABOVE each circle (no overlap with numbers)
    ax.text(-cx, R + 0.22, label_a, ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=col_a, zorder=4)
    ax.text(cx, R + 0.22, label_b, ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=col_b, zorder=4)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.6, 2.9); ax.set_aspect('equal'); ax.axis('off')


fig, ax = plt.subplots(figsize=(config.COL_FULL * 0.85, 5.4))
draw_venn(ax, len(es_only), len(ls_only), len(both),
          'gained-G4 at ES\n(n = {:,})'.format(len(gained_ES)),
          'gained-G4 at LS\n(n = {:,})'.format(len(gained_LS)), ES_COL, LS_COL)
# caption with comfortable clearance below the circles
ax.text(0, -2.35, 'ES-only ({:,}) = gained then LOST by LS (transient)\n'
        'LS-only ({:,}) = late-acquired   ·   shared ({:,}) = gained early, retained'.format(
            len(es_only), len(ls_only), len(both)),
        ha='center', va='top', fontsize=10, color='#555', zorder=4)
ax.set_title('G4-gained promoters: ES vs LS\n(gained = G4− at NG → G4+ at stage)',
             fontsize=13, fontweight='bold')
config.panel_label(ax, 'M')
fig.savefig(PNG / 'fig3M_gained_ES_LS_venn.png', dpi=300, bbox_inches='tight', facecolor='white')
sv = SVG / 'fig3M_gained_ES_LS_venn.svg'
fig.savefig(sv, format='svg', bbox_inches='tight', facecolor='white')
config.fix_svg_file(sv)
plt.close(fig)
print('Venn: ES_gained={:,} LS_gained={:,} both={:,} ES_only={:,} LS_only={:,}'.format(
    len(gained_ES), len(gained_LS), len(both), len(es_only), len(ls_only)))


# ---- M2: GO of ES-only & LS-only (vs all genes) ----------------------------
def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(1, n + 1)))[::-1])[::-1]
    return np.clip(q, 0, 1)


def enrich(query, gene_sets, bg):
    q = set(query) & bg; N = len(bg); nq = len(q)
    rows = []
    for term, gs in gene_sets.items():
        gsb = gs & bg; m = len(gsb)
        if m < 5: continue
        k = len(q & gsb)
        if k == 0: continue
        orr, p = fisher_exact([[k, nq - k], [m - k, N - nq - (m - k)]])
        rows.append({'term': term, 'overlap': k, 'odds_ratio': orr,
                     'enriched': k >= nq * m / N, 'p_raw': p, 'genes': ';'.join(sorted(q & gsb))})
    d = pd.DataFrame(rows)
    if d.empty: return d
    d['q_fdr'] = bh_fdr(d.p_raw.values)
    d['p_raw'] = d.p_raw.clip(lower=MIN_P); d['q_fdr'] = d.q_fdr.clip(lower=MIN_P)
    d['log10_p_raw'] = np.log10(d.p_raw); d['log10_q_fdr'] = np.log10(d.q_fdr)
    return d


SETS = [('ES-only\n(transient)', es_only), ('LS-only\n(late)', ls_only)]
master = []
for ont_key, gmt in ONTOLOGIES.items():
    gs = config.load_gmt(str(GMT_DIR / gmt))
    per = {lbl: enrich(s, gs, all_genes) for lbl, s in SETS}
    # rows = union of top-N sig enriched per set, ranked by ES-only log2 OR
    rowset = []
    for lbl, _ in SETS:
        d = per[lbl]
        if d.empty: continue
        rowset += d[(d.q_fdr < FDR_THR) & (d.enriched)].nsmallest(TOP_N, 'q_fdr').term.tolist()
    rows = list(dict.fromkeys(rowset))
    if not rows:
        print(f'  [no sig] {ont_key}'); continue
    first = per[SETS[0][0]].set_index('term')
    rows = sorted(rows, key=lambda t: np.log2(first.odds_ratio.get(t, np.nan))
                  if (t in first.index and first.odds_ratio.get(t, 0) > 0) else -np.inf, reverse=True)
    Meff = np.zeros((len(rows), 2)); Mq = np.full((len(rows), 2), np.nan)
    for ci, (lbl, _) in enumerate(SETS):
        d = per[lbl].set_index('term')
        for ri, t in enumerate(rows):
            if t in d.index:
                orr = d.odds_ratio.get(t, np.nan)
                Meff[ri, ci] = float(np.clip(np.log2(orr), -LOG2_CAP, LOG2_CAP)) if (orr and orr > 0 and np.isfinite(orr)) else (LOG2_CAP if np.isinf(orr) else 0.0)
                Mq[ri, ci] = d.q_fdr.get(t, np.nan)
    # wide enough for long term names; wrap long terms onto 2 lines (no truncation).
    import textwrap
    def wrap(t):
        t = t.replace(' (GO:', '\n(GO:')  # GO id on its own line
        return '\n'.join(textwrap.wrap(t, width=42, max_lines=2, placeholder='…'))
    wrapped = [wrap(r) for r in rows]
    n_extra = sum(lbl.count('\n') for lbl in wrapped)   # extra height for wrapped rows
    fig, ax = plt.subplots(figsize=(config.COL_FULL * 1.2, 0.44 * len(rows) + 0.10 * n_extra + 2.0))
    im = config.effect_heatmap(ax, Meff, wrapped, ['ES-only\n(transient)', 'LS-only\n(late)'], star_q=Mq)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=11)
    ax.set_aspect('auto')
    cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.02, aspect=18)
    cb.set_label('log2(odds ratio) vs all genes', fontsize=10)
    ax.set_title('Timing-unique gained-G4 promoters · {}\n'
                 '* q<0.05  ** q<0.01  *** q<0.001  **** q<1e-10'.format(ont_key),
                 fontsize=12, fontweight='bold')
    config.panel_label(ax, 'M')
    fig.subplots_adjust(left=0.50, right=0.88, top=0.92, bottom=0.07)
    fig.savefig(PNG / f'fig3M_GO_ESonly_LSonly_{ont_key}.png', dpi=300, bbox_inches='tight', facecolor='white')
    sv = SVG / f'fig3M_GO_ESonly_LSonly_{ont_key}.svg'
    fig.savefig(sv, format='svg', bbox_inches='tight', facecolor='white')
    config.fix_svg_file(sv)
    plt.close(fig)
    for lbl, _ in SETS:
        d = per[lbl].copy(); d.insert(0, 'set', lbl.split('\n')[0]); d.insert(0, 'ontology', ont_key)
        master.append(d)
    print(f'  {ont_key}: {len(rows)} rows')
if master:
    pd.concat(master, ignore_index=True).to_csv(DATA / 'fig3M_GO_ESonly_LSonly_master.csv', index=False)
print('done — Venn + GO of ES-only / LS-only gained-G4 promoters')
