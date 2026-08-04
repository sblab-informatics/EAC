#!/usr/bin/env python3
"""
Figure 3 — Panel E: G4 status of EAC driver-gene promoters across progression.

Named esophageal-adenocarcinoma (EAC)/Barrett's driver genes, showing the G4
status of their promoter at NG / ES_EAC / LS_EAC (corrected reproducible BG4),
their chromatin state, and their expression — so the chromatin-architecture
finding (G4-gain) maps onto the genes a clinician knows.

Drivers grouped as oncogenes (OG) vs tumour suppressors (TSG). For each: a
G4-status strip (NG, ES, LS; G4+ red / G4- blue), a 'gained G4' flag, and the
per-stage TPM. Effect/colour are categorical (G4 status); expression shown as a
small log2-TPM track. Raw data saved.

Driver list = canonical EAC/Barrett's drivers (TCGA-ESCA; Dulak 2013; Frankell
2019; Stachler 2015). Source: data/promoter_g4_status_corrected.csv,
data/promoter_histone_marks_corrected.csv, data/rnaseq/.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA_DIR = SCRIPT_DIR / 'data'
PNG_DIR = SCRIPT_DIR / 'figures' / 'png'
SVG_DIR = SCRIPT_DIR / 'figures' / 'svg'
config.setup_style()

STAGES = ['NG', 'ES_EAC', 'LS_EAC']
STAGE_LAB = ['NG', 'ES', 'LS']
G4P, G4N = config.G4_COLORS['G4+'], config.G4_COLORS['G4-']

# canonical EAC/Barrett's drivers, classified OG / TSG (literature)
DRIVERS = {
    'TP53': 'TSG', 'CDKN2A': 'TSG', 'SMAD4': 'TSG', 'ARID1A': 'TSG', 'RB1': 'TSG',
    'APC': 'TSG', 'PTEN': 'TSG', 'FHIT': 'TSG', 'WWOX': 'TSG', 'RNF43': 'TSG',
    'TGFBR2': 'TSG', 'SMARCA4': 'TSG', 'NOTCH1': 'TSG',
    'ERBB2': 'OG', 'KRAS': 'OG', 'MYC': 'OG', 'CCNE1': 'OG', 'CCND1': 'OG',
    'EGFR': 'OG', 'MET': 'OG', 'FGFR2': 'OG', 'PIK3CA': 'OG', 'CDK6': 'OG',
    'MDM2': 'OG', 'AKT1': 'OG', 'BRAF': 'OG', 'GATA4': 'OG', 'GATA6': 'OG',
    'KLF5': 'OG', 'VEGFA': 'OG', 'SOX9': 'OG', 'ELMO1': 'OG', 'CTNNB1': 'OG',
}

g4 = pd.read_csv(DATA_DIR / 'promoter_g4_status_corrected.csv').set_index('gene')
tpm = pd.read_csv(DATA_DIR / 'rnaseq' / 'gene_tpm_matrix.csv')
meta = pd.read_csv(DATA_DIR / 'rnaseq' / 'sample_metadata_corrected.csv')
gcol = 'gene_name' if 'gene_name' in tpm.columns else tpm.columns[0]
sm = {st: tpm.set_index(gcol)[[s for s in meta[meta.analytical_condition == st].sample_id if s in tpm.columns]].mean(axis=1)
      for st in STAGES}

rows = []
for gene, cls in DRIVERS.items():
    if gene not in g4.index:
        continue
    r = g4.loc[gene]
    status = [r[f'g4_{s}'] for s in STAGES]
    gained = (status[0] == 'G4-') and ('G4+' in status[1:])
    rows.append({'gene': gene, 'class': cls,
                 'g4_NG': status[0], 'g4_ES_EAC': status[1], 'g4_LS_EAC': status[2],
                 'gained_G4': gained,
                 'tpm_NG': round(sm['NG'].get(gene, np.nan), 2),
                 'tpm_ES_EAC': round(sm['ES_EAC'].get(gene, np.nan), 2),
                 'tpm_LS_EAC': round(sm['LS_EAC'].get(gene, np.nan), 2)})
df = pd.DataFrame(rows)
# order: gained first (then stable G4+, then G4-), within class
df['order_key'] = df.apply(lambda x: (0 if x.gained_G4 else (1 if x.g4_NG == 'G4+' else 2)), axis=1)
df = df.sort_values(['class', 'order_key', 'gene']).reset_index(drop=True)
df.to_csv(DATA_DIR / 'fig3E_driver_g4_status_rawdata.csv', index=False)

# ---- plot: G4-status grid + gained flag + log2 TPM track -------------------
n = len(df)
fig, (axg, axt) = plt.subplots(1, 2, figsize=(config.COL_FULL, 0.34 * n + 1.5),
                               gridspec_kw={'width_ratios': [3, 3], 'wspace': 0.05})

# left: G4 status grid (3 stage columns) coloured red/blue, with gained marker
for i, (_, r) in enumerate(df.iterrows()):
    for j, s in enumerate(STAGES):
        col = G4P if r[f'g4_{s}'] == 'G4+' else G4N
        axg.add_patch(plt.Rectangle((j, n - 1 - i - 0.45), 0.9, 0.9, color=col, ec='white', lw=0.6))
    if r.gained_G4:
        axg.text(3.05, n - 1 - i, '▲ gained', va='center', ha='left', fontsize=10,
                 color=G4P, fontweight='bold')
axg.set_xlim(-0.1, 4.2); axg.set_ylim(-0.6, n - 0.4)
axg.set_xticks([0.45, 1.45, 2.45]); axg.set_xticklabels(STAGE_LAB, fontsize=12)
axg.set_yticks(range(n)); axg.set_yticklabels(df['gene'][::-1].tolist(), fontsize=10)
# class separators / labels via y tick colour
for i, (_, r) in enumerate(df.iterrows()):
    axg.get_yticklabels()[n - 1 - i].set_color('#b2182b' if r['class'] == 'OG' else '#2166ac')
axg.set_title('Promoter G4 status', fontsize=12, fontweight='bold')
for sp in axg.spines.values():
    sp.set_visible(False)
axg.tick_params(length=0)

# right: log2(TPM+1) track across stages
T = np.log2(df[[f'tpm_{s}' for s in STAGES]].fillna(0).values + 1)
import matplotlib.colors as mcolors
im = axt.imshow(T[::-1], aspect='auto', cmap='Greys',
                extent=[-0.5, 2.5, -0.5, n - 0.5])
axt.set_xticks([0, 1, 2]); axt.set_xticklabels(STAGE_LAB, fontsize=12)
axt.set_yticks([]); axt.set_title('log2(TPM+1)', fontsize=12, fontweight='bold')
cbar = fig.colorbar(im, ax=axt, shrink=0.4, pad=0.03)
cbar.set_label('log2(TPM+1)', fontsize=10)

# legend
leg = [Patch(facecolor=G4P, label='G4+'), Patch(facecolor=G4N, label='G4−')]
fig.legend(handles=leg, loc='lower center', ncol=2, frameon=False, fontsize=12,
           bbox_to_anchor=(0.3, -0.02))
fig.text(0.5, 1.0, 'EAC driver-gene promoters: G4 status & expression across progression\n'
         'gene labels: red = oncogene, blue = tumour suppressor · ▲ = gains promoter G4',
         ha='center', va='bottom', fontsize=12, fontweight='bold')
config.panel_label(axg, 'E')
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(PNG_DIR / 'fig3E_driver_g4_status.png', dpi=300, bbox_inches='tight', facecolor='white')
svg = SVG_DIR / 'fig3E_driver_g4_status.svg'
fig.savefig(svg, format='svg', bbox_inches='tight', facecolor='white')
with open(svg) as fh:
    t = fh.read()
with open(svg, 'w') as fh:
    fh.write(config.fix_svg_for_affinity(t))
plt.close(fig)
print(f'saved fig3E_driver_g4_status ({n} drivers; {df.gained_G4.sum()} gain a promoter G4)')
