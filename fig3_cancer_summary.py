#!/usr/bin/env python3
"""
Figure 3 — Panel D: cancer-oriented summary of G4-gain at promoters (EAC).

A CURATED, cross-ontology heatmap of the cancer-progression programmes whose
promoters gain a G4 during esophageal adenocarcinoma (EAC) progression. Terms
are hand-picked from the significant gained-G4 enrichments (Panel C master
table) and grouped into cancer hallmarks relevant to Barrett's->EAC.

Columns = the gain transitions (NG->ES, NG->LS, NG->ES/LS). Colour = log2(odds
ratio) [effect size, RdBu_r], stars = FDR q. Source = the PRIMARY analysis
(V1: gained vs all genes — the biological 'where does G4-gain land' question).
Reads data/fig3C_GO_master_results.csv.

This is a focused, reviewer-friendly EAC narrative panel; the full 25 heatmaps
(Panel C) remain the exhaustive supplement.
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

DATA_DIR = SCRIPT_DIR / 'data'
PNG_DIR = SCRIPT_DIR / 'figures' / 'png'
SVG_DIR = SCRIPT_DIR / 'figures' / 'svg'
config.setup_style()

COLS = ['NG_to_ES', 'NG_to_LS', 'NG_to_either']
COL_LABEL = {'NG_to_ES': 'NG→ES', 'NG_to_LS': 'NG→LS', 'NG_to_either': 'NG→ES/LS'}

# Curated cancer programmes (label -> (ontology, exact term)). Picked from the
# significant V1 gained-G4 enrichments; grouped by EAC-relevant hallmark.
CURATED = [
    # --- p53 / tumour suppressor ---
    ('p53 pathway (Hallmark)',                 'Hallmark',  'p53 Pathway'),
    ('p53 signaling (KEGG)',                   'KEGG',      'P53 SIGNALING PATHWAY'),
    ('p53-repressed targets up (Oncogenic)',   'Oncogenic', 'P53 DN.V1 UP'),
    ('RB1–E2F defective binding (Reactome)',   'Reactome',  'Defective Binding of RB1 Mutants to E2F1,(E2F2, E2F3)'),
    # --- cell cycle / proliferation ---
    ('G2–M checkpoint (Hallmark)',             'Hallmark',  'G2-M Checkpoint'),
    ('Mitotic spindle (Hallmark)',             'Hallmark',  'Mitotic Spindle'),
    ('E2F1 targets up (Oncogenic)',            'Oncogenic', 'E2F1 UP.V1 UP'),
    ('E2F DNA replication (Reactome)',         'Reactome',  'E2F Mediated Regulation of DNA Replication'),
    ('Diseases of mitotic cell cycle (Reactome)', 'Reactome', 'Diseases of Mitotic Cell Cycle'),
    # --- genome maintenance ---
    ('DNA repair (Hallmark)',                  'Hallmark',  'DNA Repair'),
    # --- metabolism (Warburg) ---
    ('Glycolysis (Hallmark)',                  'Hallmark',  'Glycolysis'),
    ('Glycolysis/gluconeogenesis (KEGG)',      'KEGG',      'GLYCOLYSIS / GLUCONEOGENESIS'),
    # --- invasion / EMT ---
    ('TGF-β signaling (Hallmark)',             'Hallmark',  'TGF-beta Signaling'),
    ('Mesenchymal cell proliferation (GO BP)', 'GO_BP',     'Positive Regulation of Mesenchymal Cell Proliferation (GO:0002053)'),
    # --- inflammation / immune ---
    ('Interferon-α response (Hallmark)',       'Hallmark',  'Interferon Alpha Response'),
    ('TNF signaling (KEGG)',                   'KEGG',      'TNF SIGNALING PATHWAY'),
    # --- other oncogenic signalling ---
    ('Hedgehog signaling (KEGG)',              'KEGG',      'HEDGEHOG SIGNALING PATHWAY'),
    ('Estrogen signaling (KEGG)',              'KEGG',      'ESTROGEN SIGNALING PATHWAY'),
]
BG = 'V1_gained_vs_allgenes'

m = pd.read_csv(DATA_DIR / 'fig3C_GO_master_results.csv')
m = m[(m.background == BG) & (m.method == 'ORA_Fisher')]

rows, Meff, Mq, found = [], [], [], []
for label, ont, term in CURATED:
    sub = m[(m.ontology == ont) & (m.term == term)]
    if sub.empty:
        print(f'  [missing] {ont} :: {term}')
        continue
    eff_row, q_row = [], []
    for col in COLS:
        c = sub[sub.transition == col]
        if c.empty:
            eff_row.append(0.0); q_row.append(np.nan)
        else:
            orr = float(c['effect'].iloc[0])
            eff_row.append(np.log2(orr) if orr > 0 else 0.0)
            q_row.append(float(c['q_fdr'].iloc[0]))
    rows.append(label); Meff.append(eff_row); Mq.append(q_row)
    found.append((label, ont, term))

Meff = np.array(Meff); Mq = np.array(Mq)

fig, ax = plt.subplots(figsize=(config.COL_FULL * 0.95, 0.42 * len(rows) + 1.8))
im = config.effect_heatmap(ax, Meff, rows, [COL_LABEL[c] for c in COLS], star_q=Mq)
cbar = fig.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
cbar.set_label('log2(odds ratio)\nG4-gain enrichment vs all genes', fontsize=11)
ax.set_title('Cancer programmes gaining promoter G4 during EAC progression\n'
             '(gained-G4 vs all genes; * q<0.05  ** q<0.01  *** q<0.001  **** q<1e-10)',
             fontsize=12, fontweight='bold')
config.panel_label(ax, 'D')
fig.tight_layout()
fig.savefig(PNG_DIR / 'fig3D_cancer_hallmark_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
svg = SVG_DIR / 'fig3D_cancer_hallmark_summary.svg'
fig.savefig(svg, format='svg', bbox_inches='tight', facecolor='white')
with open(svg) as fh:
    t = fh.read()
with open(svg, 'w') as fh:
    fh.write(config.fix_svg_for_affinity(t))
plt.close(fig)

# raw data for this curated panel
out = []
for (label, ont, term), eff, q in zip(found, Meff, Mq):
    for ci, col in enumerate(COLS):
        out.append({'panel_label': label, 'ontology': ont, 'term': term,
                    'transition': col, 'log2_odds_ratio': eff[ci], 'q_fdr': q[ci]})
pd.DataFrame(out).to_csv(DATA_DIR / 'fig3D_cancer_hallmark_summary_rawdata.csv', index=False)
print(f'saved fig3D_cancer_hallmark_summary ({len(rows)} curated cancer programmes)')
