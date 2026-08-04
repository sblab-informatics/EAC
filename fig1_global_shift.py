#!/usr/bin/env python3
"""
Figure 1 — Panel G: global G4 signal shift per stage contrast, nDR vs CPM.

One summary view of the differential: median log2 fold-change of BG4 G4 signal
across the universe for each stage contrast, under the two normalisations
(Drosophila spike-in nDR vs CPM). Shows that the genome-wide shift is captured
by spike-in normalisation and largely erased by CPM. Descriptive (effect size
only); per-region significance is in the differential panels/CSVs.

Reads data/fig1_differential_summary.csv (written by fig1_differential.py).
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

LAB = {'NG': 'NG', 'ES_EAC': 'ES', 'LS_EAC': 'LS', 'ES_Rx': 'ES+Rx', 'LS_Rx': 'LS+Rx'}
# progression contrasts (the biologically ordered ones) to show, in order
SHOW = ['NG_vs_ES_EAC', 'ES_EAC_vs_LS_EAC', 'NG_vs_LS_EAC']
NORM_STYLE = {'nDR': ('Drosophila spike-in (nDR)', '#B2182B', 'o'),
              'cpm': ('CPM', '#4393C3', 's')}


def clabel(c):
    a, b = c.split('_vs_')
    return '{} → {}'.format(LAB[a], LAB[b])


summ = pd.read_csv(DATA_DIR / 'fig1_differential_summary.csv')

fig, ax = plt.subplots(figsize=(config.COL_SINGLE * 1.5, 4.4))
x = np.arange(len(SHOW))
off = {'nDR': -0.13, 'cpm': +0.13}
for norm, (label, col, mk) in NORM_STYLE.items():
    ys = []
    for c in SHOW:
        row = summ[(summ['norm'] == norm) & (summ['contrast'] == c)]
        ys.append(float(row['median_log2FC'].iloc[0]) if len(row) else np.nan)
    ax.plot(x + off[norm], ys, marker=mk, ms=12, lw=1.6, color=col,
            markeredgecolor='white', markeredgewidth=1.0, label=label, zorder=3)
    for xi, y in zip(x + off[norm], ys):
        ax.annotate('{:+.2f}'.format(y), (xi, y), xytext=(0, 8 if norm == 'nDR' else -14),
                    textcoords='offset points', ha='center', fontsize=11,
                    color=col, fontweight='bold')

ax.axhline(0, color='#333', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([clabel(c) for c in SHOW])
ax.set_ylabel('median log2 FC of G4 signal\n(across G4 universe)')
ax.set_title('Global G4 signal shift across progression', fontsize=14)
ax.legend(frameon=False, fontsize=12, loc='upper left')
ax.set_ylim(-0.2, max(0.2, summ[summ['contrast'].isin(SHOW)]['median_log2FC'].max() * 1.25))
config.panel_label(ax, 'G')
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(PNG_DIR / 'fig1_G_global_shift_nDR_vs_cpm.png', dpi=300,
            bbox_inches='tight', facecolor='white')
svg_path = SVG_DIR / 'fig1_G_global_shift_nDR_vs_cpm.svg'
fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
with open(svg_path) as fh:
    txt = fh.read()
with open(svg_path, 'w') as fh:
    fh.write(config.fix_svg_for_affinity(txt))
plt.close(fig)

# tidy data table for this panel
summ[summ['contrast'].isin(SHOW)].to_csv(DATA_DIR / 'fig1_G_global_shift.csv', index=False)
print('saved fig1_G_global_shift_nDR_vs_cpm')
print(summ[summ['contrast'].isin(SHOW)][['norm', 'contrast', 'median_log2FC',
      'n_up_in_B', 'n_down_in_B', 'testable_donor_level']].to_string(index=False))
