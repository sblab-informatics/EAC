#!/usr/bin/env python3
"""
Figure 3 — Panel N: Venn of (G4+ AND expressed) promoters, ES vs LS, + GO of EACH
of the three Venn regions (ES-only, shared, LS-only).

Each circle = promoters that are BOTH G4-positive AND expressed at a stage:
  ES set : G4(ES) positive AND TPM(ES) > 0.1
  LS set : G4(LS) positive AND TPM(LS) > 0.1

The Venn partitions these into three regions, each profiled by GO:
  ES-only  — G4+/expressed at ES but NOT at LS  (early-restricted)
  shared   — G4+/expressed at BOTH ES and LS    (sustained)
  LS-only  — G4+/expressed at LS but NOT at ES   (late-restricted)

Two orthogonal design axes are run separately (4 Venns total), per request:
  g4_mode  ∈ {presence, gained}
    presence : g4_<stage> == 'G4+'                       (where are expressed G4+ proms)
    gained   : g4_NG == 'G4-' AND g4_<stage> == 'G4+'     (acquisition, expressed)
  tpm_mode ∈ {match, any}
    match    : TPM > 0.1 at the MATCHING stage (ES set uses ES TPM, LS set uses LS TPM)
    any      : TPM > 0.1 in ANY of NG/ES/LS (same expressed universe for both sets)

GO: 5 ontologies (GO BP 2025, KEGG 2026, Reactome 2024, MSigDB Hallmark 2020,
MSigDB Oncogenic), Fisher exact vs ALL genes (chromatin framing — background is
the full promoter gene universe, NOT expression-matched), effect-size led
(log2 OR heatmap + BH-FDR stars). p bounded (1e-300 floor + log10 companion);
unit = promoters (NOT donors). Every Venn + every GO heatmap writes raw data.
"""
import sys
import textwrap
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
RNA_DIR = DATA / 'rnaseq'
PNG = SCRIPT_DIR / 'figures' / 'png'
SVG = SCRIPT_DIR / 'figures' / 'svg'
GMT_DIR = SCRIPT_DIR / 'reference_data' / 'gene_sets'
config.setup_style()

MIN_P = 1e-300
TOP_N = 15
FDR_THR = 0.05
LOG2_CAP = 6.0
TPM_THR = 0.1
STAGES = ['NG', 'ES_EAC', 'LS_EAC']
ONTOLOGIES = {'GO_BP': 'go_biological_process_2025.gmt', 'KEGG': 'kegg_2026.gmt',
              'Reactome': 'reactome_pathways_2024.gmt', 'Hallmark': 'msigdb_hallmark_2020.gmt',
              'Oncogenic': 'msigdb_oncogenic_signatures.gmt'}
ES_COL, LS_COL = '#F4A582', '#B2182B'   # ES / LS condition colours
SHARE_COL = '#762A83'                     # shared region accent (purple, distinct)


# ---- load promoters + per-stage TPM (identical recipe to fig3_histone_state_flow) -
g = pd.read_csv(DATA / 'promoter_g4_status_corrected.csv')
tpm = pd.read_csv(RNA_DIR / 'gene_tpm_matrix.csv')
meta = pd.read_csv(RNA_DIR / 'sample_metadata_corrected.csv')
gcol = 'gene_name' if 'gene_name' in tpm.columns else tpm.columns[0]
for st in STAGES:
    sids = [s for s in meta[meta['analytical_condition'] == st]['sample_id'] if s in tpm.columns]
    g[f'tpm_{st}'] = g['gene'].map(tpm.set_index(gcol)[sids].mean(axis=1))
tpm_cols = [f'tpm_{s}' for s in STAGES]
g['tpm_any01'] = (g[tpm_cols] > TPM_THR).any(axis=1)
all_genes = set(g.gene)
print('universe {:,} promoters | TPM>{} any-stage {:,}'.format(
    len(g), TPM_THR, int(g.tpm_any01.sum())))


# ---- GO machinery ----------------------------------------------------------
def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(1, n + 1)))[::-1])[::-1]
    return np.clip(q, 0, 1)


def enrich(query, gene_sets, bg):
    q = set(query) & bg; N = len(bg); nq = len(q)
    rows = []
    if nq == 0:
        return pd.DataFrame(rows)
    for term, gs in gene_sets.items():
        gsb = gs & bg; m = len(gsb)
        if m < 5: continue
        k = len(q & gsb)
        if k == 0: continue
        orr, p = fisher_exact([[k, nq - k], [m - k, N - nq - (m - k)]])
        rows.append({'term': term, 'overlap': k, 'query_n': nq, 'set_n': m,
                     'odds_ratio': orr, 'enriched': k >= nq * m / N, 'p_raw': p,
                     'genes': ';'.join(sorted(q & gsb))})
    d = pd.DataFrame(rows)
    if d.empty: return d
    d['q_fdr'] = bh_fdr(d.p_raw.values)
    d['p_raw'] = d.p_raw.clip(lower=MIN_P); d['q_fdr'] = d.q_fdr.clip(lower=MIN_P)
    d['log10_p_raw'] = np.log10(d.p_raw); d['log10_q_fdr'] = np.log10(d.q_fdr)
    return d


def wrap_term(t):
    t = t.replace(' (GO:', '\n(GO:')
    return '\n'.join(textwrap.wrap(t, width=42, max_lines=2, placeholder='…'))


# ---- Venn (clean: equal radii, controlled overlap, no label clash) ----------
def draw_venn(ax, only_a, only_b, both_n, label_a, label_b, col_a, col_b):
    R = 1.45
    cx = 0.95
    ca, cb = (-cx, 0), (cx, 0)
    ax.add_patch(Circle(ca, R, fc=col_a, ec=col_a, lw=2.0, alpha=0.40, zorder=2))
    ax.add_patch(Circle(cb, R, fc=col_b, ec=col_b, lw=2.0, alpha=0.40, zorder=2))
    ax.text(-cx - 0.55, 0, '{:,}'.format(only_a), ha='center', va='center',
            fontsize=17, fontweight='bold', color=col_a, zorder=4)
    ax.text(cx + 0.55, 0, '{:,}'.format(only_b), ha='center', va='center',
            fontsize=17, fontweight='bold', color=col_b, zorder=4)
    ax.text(0, 0, '{:,}'.format(both_n), ha='center', va='center',
            fontsize=17, fontweight='bold', color='#222', zorder=4)
    ax.text(-cx, R + 0.22, label_a, ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=col_a, zorder=4)
    ax.text(cx, R + 0.22, label_b, ha='center', va='bottom', fontsize=12,
            fontweight='bold', color=col_b, zorder=4)
    ax.set_xlim(-3.4, 3.4); ax.set_ylim(-2.6, 2.9); ax.set_aspect('equal'); ax.axis('off')


# ---- one full variant (Venn + 3-region GO across 5 ontologies) -------------
def run_variant(g4_mode, tpm_mode):
    tag = '{}_{}'.format(g4_mode, tpm_mode)            # e.g. presence_match
    # --- masks --------------------------------------------------------------
    if g4_mode == 'presence':
        es_g4 = g.g4_ES_EAC == 'G4+'
        ls_g4 = g.g4_LS_EAC == 'G4+'
        g4_desc = 'G4+ at stage'
    else:  # gained
        es_g4 = (g.g4_NG == 'G4-') & (g.g4_ES_EAC == 'G4+')
        ls_g4 = (g.g4_NG == 'G4-') & (g.g4_LS_EAC == 'G4+')
        g4_desc = 'G4-gained (G4− NG → G4+ stage)'
    if tpm_mode == 'match':
        es_expr = g.tpm_ES_EAC > TPM_THR
        ls_expr = g.tpm_LS_EAC > TPM_THR
        tpm_desc = 'TPM > {} at matching stage'.format(TPM_THR)
    else:  # any
        es_expr = g.tpm_any01
        ls_expr = g.tpm_any01
        tpm_desc = 'TPM > {} in any stage'.format(TPM_THR)

    es_set = set(g[es_g4 & es_expr].gene)
    ls_set = set(g[ls_g4 & ls_expr].gene)
    both = es_set & ls_set
    es_only = es_set - ls_set
    ls_only = ls_set - es_set

    # --- persist set membership --------------------------------------------
    pd.DataFrame([
        {'set': 'ES_g4expr', 'n': len(es_set)}, {'set': 'LS_g4expr', 'n': len(ls_set)},
        {'set': 'ES_and_LS_shared', 'n': len(both)},
        {'set': 'ES_only', 'n': len(es_only)}, {'set': 'LS_only', 'n': len(ls_only)},
        {'set': 'g4_mode', 'n': g4_mode}, {'set': 'tpm_mode', 'n': tpm_mode},
    ]).to_csv(DATA / 'fig3N_venn_{}_rawdata.csv'.format(tag), index=False)
    for nm, s in [('ES_only', es_only), ('shared', both), ('LS_only', ls_only)]:
        pd.DataFrame({'gene': sorted(s)}).to_csv(
            DATA / 'fig3N_{}_{}_genes.csv'.format(tag, nm), index=False)

    # --- Venn figure --------------------------------------------------------
    fig, ax = plt.subplots(figsize=(config.COL_FULL * 0.85, 5.6))
    draw_venn(ax, len(es_only), len(ls_only), len(both),
              'ES: G4+ & expressed\n(n = {:,})'.format(len(es_set)),
              'LS: G4+ & expressed\n(n = {:,})'.format(len(ls_set)), ES_COL, LS_COL)
    ax.text(0, -2.35,
            'ES-only ({:,}) = G4+/expressed at ES, not LS   ·   shared ({:,}) = both stages\n'
            'LS-only ({:,}) = G4+/expressed at LS, not ES'.format(
                len(es_only), len(both), len(ls_only)),
            ha='center', va='top', fontsize=10, color='#555', zorder=4)
    ax.set_title('Expressed G4+ promoters: ES vs LS\n{} · {}'.format(g4_desc, tpm_desc),
                 fontsize=12.5, fontweight='bold')
    config.panel_label(ax, 'N')
    fig.savefig(PNG / 'fig3N_venn_{}.png'.format(tag), dpi=300, bbox_inches='tight', facecolor='white')
    sv = SVG / 'fig3N_venn_{}.svg'.format(tag)
    fig.savefig(sv, format='svg', bbox_inches='tight', facecolor='white')
    config.fix_svg_file(sv)
    plt.close(fig)
    print('  [{}] Venn ES={:,} LS={:,} | ES-only={:,} shared={:,} LS-only={:,}'.format(
        tag, len(es_set), len(ls_set), len(es_only), len(both), len(ls_only)))

    # --- GO of the 3 regions ------------------------------------------------
    SETS = [('ES-only', es_only), ('shared', both), ('LS-only', ls_only)]
    col_labels = ['ES-only\n(early)', 'shared\n(sustained)', 'LS-only\n(late)']
    master = []
    for ont_key, gmt in ONTOLOGIES.items():
        gs = config.load_gmt(str(GMT_DIR / gmt))
        per = {lbl: enrich(s, gs, all_genes) for lbl, s in SETS}
        # rows = union of top-N sig-enriched per region
        rowset = []
        for lbl, _ in SETS:
            d = per[lbl]
            if d.empty: continue
            rowset += d[(d.q_fdr < FDR_THR) & (d.enriched)].nsmallest(TOP_N, 'q_fdr').term.tolist()
        rows = list(dict.fromkeys(rowset))
        # accumulate master regardless of whether anything plots
        for lbl, _ in SETS:
            d = per[lbl]
            if d.empty: continue
            d = d.copy(); d.insert(0, 'region', lbl); d.insert(0, 'ontology', ont_key)
            d.insert(0, 'variant', tag)
            master.append(d)
        if not rows:
            print('    [no sig] {} {}'.format(tag, ont_key)); continue
        # order rows by shared-region log2 OR (fall back to ES-only, then LS-only), descending.
        # empty-safe: enrich() returns a column-less frame for an empty region, so guard set_index.
        _ref_src = next((per[k] for k in ('shared', 'ES-only', 'LS-only') if not per[k].empty),
                        pd.DataFrame(columns=['term', 'odds_ratio']))
        ref = _ref_src.set_index('term')
        rows = sorted(rows, key=lambda t: np.log2(ref.odds_ratio.get(t, np.nan))
                      if (t in ref.index and ref.odds_ratio.get(t, 0) > 0) else -np.inf, reverse=True)
        Meff = np.zeros((len(rows), 3)); Mq = np.full((len(rows), 3), np.nan)
        for ci, (lbl, _) in enumerate(SETS):
            d = per[lbl].set_index('term') if not per[lbl].empty else pd.DataFrame().set_index(pd.Index([]))
            for ri, t in enumerate(rows):
                if t in d.index:
                    orr = d.odds_ratio.get(t, np.nan)
                    Meff[ri, ci] = (float(np.clip(np.log2(orr), -LOG2_CAP, LOG2_CAP))
                                    if (orr and orr > 0 and np.isfinite(orr))
                                    else (LOG2_CAP if np.isinf(orr) else 0.0))
                    Mq[ri, ci] = d.q_fdr.get(t, np.nan)
        wrapped = [wrap_term(r) for r in rows]
        n_extra = sum(lbl.count('\n') for lbl in wrapped)

        def draw_go(scale_mode):
            """Draw one GO heatmap. scale_mode='symmetric' = ±cap diverging (default,
            comparable across panels); 'data' = colour spans THIS heatmap's true
            min→max log2 OR, with those limits labelled at the colorbar ends."""
            fig, ax = plt.subplots(figsize=(config.COL_FULL * 1.3,
                                            0.44 * len(rows) + 0.10 * n_extra + 2.0))
            if scale_mode == 'data':
                im, vmin, vmax = config.effect_heatmap(ax, Meff, wrapped, col_labels,
                                                       star_q=Mq, scale='data')
                suffix, fname = '_datascaled', 'fig3N_GO_{}_{}_datascaled'.format(tag, ont_key)
            else:
                im = config.effect_heatmap(ax, Meff, wrapped, col_labels, star_q=Mq)
                vmin, vmax, suffix, fname = None, None, '', 'fig3N_GO_{}_{}'.format(tag, ont_key)
            ax.tick_params(axis='y', labelsize=9)
            ax.tick_params(axis='x', labelsize=11)
            ax.set_aspect('auto')
            cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.02, aspect=18)
            if scale_mode == 'data':
                cb.set_ticks([vmin, (vmin + vmax) / 2, vmax])
                cb.set_ticklabels(['{:.2f}'.format(vmin), '{:.2f}'.format((vmin + vmax) / 2),
                                   '{:.2f}'.format(vmax)])
                cb.set_label('log2(odds ratio) vs all genes\n'
                             '(scale = data min {:.2f} … max {:.2f})'.format(vmin, vmax),
                             fontsize=9)
            else:
                cb.set_label('log2(odds ratio) vs all genes', fontsize=10)
            scale_note = '  ·  colour = data min→max' if scale_mode == 'data' else ''
            ax.set_title('Expressed G4+ promoters by ES/LS region · {}{}\n'
                         '{} · {}\n* q<0.05  ** q<0.01  *** q<0.001  **** q<1e-10'.format(
                             ont_key, scale_note, g4_desc, tpm_desc),
                         fontsize=11, fontweight='bold')
            config.panel_label(ax, 'N')
            fig.subplots_adjust(left=0.50, right=0.88, top=0.90, bottom=0.07)
            fig.savefig(PNG / (fname + '.png'), dpi=300, bbox_inches='tight', facecolor='white')
            sv = SVG / (fname + '.svg')
            fig.savefig(sv, format='svg', bbox_inches='tight', facecolor='white')
            config.fix_svg_file(sv)
            plt.close(fig)
            return vmin, vmax

        draw_go('symmetric')                        # existing panel (unchanged)
        dvmin, dvmax = draw_go('data')              # new data min→max copy
        print('    {} {}: {} rows  (datascaled colour [{:.2f}, {:.2f}])'.format(
            tag, ont_key, len(rows), dvmin, dvmax))
    if master:
        pd.concat(master, ignore_index=True).to_csv(
            DATA / 'fig3N_GO_{}_master.csv'.format(tag), index=False)
    return {'tag': tag, 'ES': len(es_set), 'LS': len(ls_set),
            'ES_only': len(es_only), 'shared': len(both), 'LS_only': len(ls_only)}


# ---- run all 4 variants ----------------------------------------------------
summary = []
for g4_mode in ['presence', 'gained']:
    for tpm_mode in ['match', 'any']:
        print('=== variant: g4={} tpm={} ==='.format(g4_mode, tpm_mode))
        summary.append(run_variant(g4_mode, tpm_mode))
pd.DataFrame(summary).to_csv(DATA / 'fig3N_all_variants_summary.csv', index=False)
print('\ndone — 4 Venns + 3-region GO each (Panel N)')
print(pd.DataFrame(summary).to_string(index=False))
