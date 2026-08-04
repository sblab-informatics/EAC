#!/usr/bin/env python3
"""
Figure 3 — TPM>0.1 heatmap variants with data-scaled colorbars.

Same gained-G4 GO/pathway heatmaps as fig3_go_gained.py (Panel C) and the
cancer summary (Panel D), but:
  1. QUERY = EXPRESSED gainers (gene TPM>0.1 in any stage) — the V2 background
     'gainedExpr_vs_allgenes' already in data/fig3C_GO_master_results.csv.
  2. COLORBAR scale = each heatmap's TRUE data range (not a fixed symmetric cap),
     with the actual min/max log2(odds ratio) labelled at the colorbar ends.
  3. Colour scheme unchanged (RdBu_r); only the SCALE differs.

Outputs (new, alongside the originals):
  fig3C_GO_gained_TPM01_<ontology>.{png,svg}   (5 ontologies)
  fig3D_cancer_hallmark_summary_TPM01.{png,svg}
  + per-figure *_rawdata.csv
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA = SCRIPT_DIR / 'data'
PNG = SCRIPT_DIR / 'figures' / 'png'
SVG = SCRIPT_DIR / 'figures' / 'svg'
config.setup_style()

COLS = ['NG_to_ES', 'NG_to_LS', 'NG_to_either']
COL_LABEL = {'NG_to_ES': 'NG→ES', 'NG_to_LS': 'NG→LS', 'NG_to_either': 'NG→ES/LS'}
ONTOLOGIES = ['GO_BP', 'KEGG', 'Reactome', 'Hallmark', 'Oncogenic']
BG = 'V2_gainedExpr_vs_allgenes'   # expressed (TPM>0.1) gainers vs all genes
TOP_N = 15
FDR_THR = 0.05

m = pd.read_csv(DATA / 'fig3C_GO_master_results.csv')
m = m[(m.background == BG) & (m.method == 'ORA_Fisher')]


def data_scaled_heatmap(Meff, Mq, rows, fname, title, cbar_label):
    """Render with scale='data' so the colorbar ends carry the true min/max."""
    h = max(3.0, 0.34 * len(rows) + 1.8)
    fig, ax = plt.subplots(figsize=(config.COL_FULL * 0.9, h))
    im, vmin, vmax = config.effect_heatmap(
        ax, Meff, [r[:48] for r in rows], [COL_LABEL[c] for c in COLS],
        star_q=Mq, scale='data')
    cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    # numeric limits at the colorbar ends = the heatmap's actual min/max
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_ticklabels(['{:.2f}'.format(vmin), '{:.2f}'.format((vmin + vmax) / 2),
                         '{:.2f}'.format(vmax)])
    cbar.set_label(cbar_label + '\n(scale = data min {:.2f} … max {:.2f})'.format(vmin, vmax),
                   fontsize=10.5)
    ax.set_title(title, fontsize=12, fontweight='bold')
    config.panel_label(ax, 'C')
    fig.tight_layout()
    fig.savefig(PNG / f'{fname}.png', dpi=300, bbox_inches='tight', facecolor='white')
    sv = SVG / f'{fname}.svg'
    fig.savefig(sv, format='svg', bbox_inches='tight', facecolor='white')
    config.fix_svg_file(sv)
    plt.close(fig)
    return vmin, vmax


def build_matrix(sub, rows):
    """log2(OR) matrix + q matrix for the given term rows × COLS."""
    Meff = np.zeros((len(rows), len(COLS))); Mq = np.full((len(rows), len(COLS)), np.nan)
    for ci, c in enumerate(COLS):
        d = sub[sub.transition == c].set_index('term')
        for ri, t in enumerate(rows):
            if t in d.index:
                orr = float(d.odds_ratio.get(t, np.nan)) if 'odds_ratio' in d else float(d.effect.get(t, np.nan))
                Meff[ri, ci] = np.clip(np.log2(orr), -6, 6) if (orr and orr > 0 and np.isfinite(orr)) else 0.0
                Mq[ri, ci] = float(d.q_fdr.get(t, np.nan))
    return Meff, Mq


# ---- per-ontology GO heatmaps (TPM>0.1 query) ------------------------------
ecol = 'odds_ratio' if 'odds_ratio' in m.columns else 'effect'
raw_rows = []
for ont in ONTOLOGIES:
    mo = m[m.ontology == ont]
    if mo.empty:
        print(f'  [no data] {ont}'); continue
    # rows = union of top-N FDR-sig ENRICHED (effect>1) terms per column,
    # ranked by NG->ES log2(effect)
    rowset = []
    for c in COLS:
        sig = mo[(mo.transition == c) & (mo.q_fdr < FDR_THR) & (mo[ecol] > 1)].nsmallest(TOP_N, 'q_fdr')
        rowset += sig.term.tolist()
    rows = list(dict.fromkeys(rowset))
    if not rows:
        print(f'  [no sig] {ont}'); continue
    first = mo[mo.transition == COLS[0]].set_index('term')
    rows = sorted(rows, key=lambda t: np.log2(first[ecol].get(t, np.nan))
                  if (t in first.index and first[ecol].get(t, 0) > 0) else -np.inf, reverse=True)
    Meff, Mq = build_matrix(mo.rename(columns={ecol: 'odds_ratio'}), rows)
    vmin, vmax = data_scaled_heatmap(
        Meff, Mq, rows, f'fig3C_GO_gained_TPM01_{ont}',
        'Gained-G4 (expressed, TPM>0.1) enrichment · {}\n'
        'query = expressed gainers vs all genes · * q<0.05 ** q<0.01 *** q<0.001 **** q<1e-10'.format(ont),
        'log2(odds ratio)')
    for ri, t in enumerate(rows):
        for ci, c in enumerate(COLS):
            raw_rows.append({'ontology': ont, 'term': t, 'transition': c,
                             'log2_odds_ratio': Meff[ri, ci], 'q_fdr': Mq[ri, ci],
                             'cbar_min': vmin, 'cbar_max': vmax})
    print(f'  {ont}: {len(rows)} rows, colorbar [{vmin:.2f}, {vmax:.2f}]')
pd.DataFrame(raw_rows).to_csv(DATA / 'fig3C_GO_gained_TPM01_rawdata.csv', index=False)

# ---- curated cancer summary (Panel D) TPM>0.1 ------------------------------
# reuse the same curated term list as fig3_cancer_summary.py
CURATED = [
    ('p53 pathway (Hallmark)', 'Hallmark', 'p53 Pathway'),
    ('p53 signaling (KEGG)', 'KEGG', 'P53 SIGNALING PATHWAY'),
    ('p53-repressed targets up (Oncogenic)', 'Oncogenic', 'P53 DN.V1 UP'),
    ('RB1–E2F defective binding (Reactome)', 'Reactome', 'Defective Binding of RB1 Mutants to E2F1,(E2F2, E2F3)'),
    ('G2–M checkpoint (Hallmark)', 'Hallmark', 'G2-M Checkpoint'),
    ('Mitotic spindle (Hallmark)', 'Hallmark', 'Mitotic Spindle'),
    ('E2F1 targets up (Oncogenic)', 'Oncogenic', 'E2F1 UP.V1 UP'),
    ('E2F DNA replication (Reactome)', 'Reactome', 'E2F Mediated Regulation of DNA Replication'),
    ('Diseases of mitotic cell cycle (Reactome)', 'Reactome', 'Diseases of Mitotic Cell Cycle'),
    ('DNA repair (Hallmark)', 'Hallmark', 'DNA Repair'),
    ('Glycolysis (Hallmark)', 'Hallmark', 'Glycolysis'),
    ('Glycolysis/gluconeogenesis (KEGG)', 'KEGG', 'GLYCOLYSIS / GLUCONEOGENESIS'),
    ('TGF-β signaling (Hallmark)', 'Hallmark', 'TGF-beta Signaling'),
    ('Mesenchymal cell proliferation (GO BP)', 'GO_BP', 'Positive Regulation of Mesenchymal Cell Proliferation (GO:0002053)'),
    ('Interferon-α response (Hallmark)', 'Hallmark', 'Interferon Alpha Response'),
    ('TNF signaling (KEGG)', 'KEGG', 'TNF SIGNALING PATHWAY'),
    ('Hedgehog signaling (KEGG)', 'KEGG', 'HEDGEHOG SIGNALING PATHWAY'),
    ('Estrogen signaling (KEGG)', 'KEGG', 'ESTROGEN SIGNALING PATHWAY'),
]
labels, Meff, Mq = [], [], []
for label, ont, term in CURATED:
    sub = m[(m.ontology == ont) & (m.term == term)]
    if sub.empty:
        print(f'  [missing] {ont} :: {term}'); continue
    er, qr = [], []
    for c in COLS:
        cc = sub[sub.transition == c]
        if cc.empty:
            er.append(0.0); qr.append(np.nan)
        else:
            orr = float(cc[ecol].iloc[0])
            er.append(np.clip(np.log2(orr), -6, 6) if orr > 0 else 0.0)
            qr.append(float(cc.q_fdr.iloc[0]))
    labels.append(label); Meff.append(er); Mq.append(qr)
Meff = np.array(Meff); Mq = np.array(Mq)
vmin, vmax = data_scaled_heatmap(
    Meff, Mq, labels, 'fig3D_cancer_hallmark_summary_TPM01',
    'Cancer programmes gaining promoter G4 — EXPRESSED gainers (TPM>0.1)\n'
    'query = expressed gainers vs all genes · * q<0.05 ** q<0.01 *** q<0.001 **** q<1e-10',
    'log2(odds ratio)')
out = []
for label, eff, q in zip(labels, Meff, Mq):
    for ci, c in enumerate(COLS):
        out.append({'panel_label': label, 'transition': c, 'log2_odds_ratio': eff[ci],
                    'q_fdr': q[ci], 'cbar_min': vmin, 'cbar_max': vmax})
pd.DataFrame(out).to_csv(DATA / 'fig3D_cancer_hallmark_summary_TPM01_rawdata.csv', index=False)
print(f'  Panel D TPM>0.1: {len(labels)} programmes, colorbar [{vmin:.2f}, {vmax:.2f}]')
print('done — TPM>0.1 heatmaps with data-scaled colorbars')
