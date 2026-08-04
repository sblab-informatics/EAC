#!/usr/bin/env python3
"""
Figure 1 — Panel C/D: PCA of BG4 G4 signal across samples.

PCA on Drosophila-spike-in-normalised (nDR) BG4 signal quantified over the
G4 universe (reproducible, manual-classification peak set). Each point is one
technical-replicate sample (33 total); coloured by stage (config.CONDITION_COLORS),
donor shown by marker shape. Two views:
  Panel C  — progression samples only (NG, ES_EAC, LS_EAC; 24 samples)
  Panel D  — all stages incl. Rx (NDBE excluded; 29 samples)

Signal source: multiBigwigSummary mean nDR signal per universe region per
sample (data/fig1_bg4_signal_over_universe.tab); universe =
_raw_data/.../G4_universe_progression.bed (20,935 peaks). Stage assignment is
the MANUAL classification (data/fig1_sample_sheet.csv).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA_DIR = SCRIPT_DIR / 'data'
PNG_DIR = SCRIPT_DIR / 'figures' / 'png'
SVG_DIR = SCRIPT_DIR / 'figures' / 'svg'
for d in (PNG_DIR, SVG_DIR):
    d.mkdir(parents=True, exist_ok=True)
config.setup_style()

# NDBE (Barrett's pre-cancer) is excluded from the PCA per the analysis design.
STAGE_ORDER = ['NG', 'ES_EAC', 'LS_EAC', 'ES_Rx', 'LS_Rx']
PROGRESSION = ['NG', 'ES_EAC', 'LS_EAC']
LABELS = {'NG': 'NG', 'ES_EAC': 'ES', 'LS_EAC': 'LS', 'ES_Rx': 'ES+Rx',
          'LS_Rx': 'LS+Rx'}
NDBE_COLOR = '#999999'
# distinct marker per donor
DONOR_MARKERS = {
    'AHM1678': 'o', 'CAM450': 's', 'CAM408': '^', 'CAM412': 'v',
    'WTSI_OESO_146': 'D', 'WTSI_OESO_117': 'P', 'CAM277': 'X',
    'CAM574': '<', 'CAM428': '>', 'CAM629': '*',
}


def stage_color(stage):
    if stage == 'NDBE':
        return NDBE_COLOR
    return config.CONDITION_COLORS.get(stage, '#888888')


def _best_legend_corner(out):
    """Pick the corner (loc string) with the fewest data points nearby, so the
    in-axes donor legend covers as little data as possible."""
    x, y = out['PC1'].values, out['PC2'].values
    xmid, ymid = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    quad = {
        'upper right': ((x > xmid) & (y > ymid)).sum(),
        'upper left':  ((x < xmid) & (y > ymid)).sum(),
        'lower right': ((x > xmid) & (y < ymid)).sum(),
        'lower left':  ((x < xmid) & (y < ymid)).sum(),
    }
    return min(quad, key=quad.get)


# Run the PCA for both normalisations: nDR (Drosophila spike-in) and CPM.
# NORMS maps a tag -> (signal table, human-readable normalisation label).
NORMS = {
    'nDR': ('fig1_bg4_signal_over_universe.tab', 'Drosophila spike-in (nDR)'),
    'cpm': ('fig1_bg4_signal_over_universe_cpm.tab', 'CPM'),
}
SHEET = pd.read_csv(DATA_DIR / 'fig1_sample_sheet.csv')


def load_matrix(sig_table):
    sig = pd.read_csv(DATA_DIR / sig_table, sep='\t')
    sig.columns = [c.strip("#'\" ") for c in sig.columns]
    meta_cols = ['chr', 'start', 'end']
    sample_cols = [c for c in sig.columns if c not in meta_cols]
    mat = np.log1p(sig[sample_cols].fillna(0.0).values.T)   # samples x regions
    lab2idx = {f"{r.sample_id}_{r.donor}_{r.stage}_{r.trep}": i
               for i, r in SHEET.iterrows()}
    order = [lab2idx[c] for c in sample_cols]
    sheet = SHEET.iloc[order].reset_index(drop=True)
    assert len(sheet) == mat.shape[0]
    return mat, sheet


def run_pca_panel(mat, sheet, norm_label, keep_stages, fname, title, panel_letter):
    keep = sheet['stage'].isin(keep_stages).values
    X = mat[keep]
    sub = sheet[keep].reset_index(drop=True)
    # keep regions with variance
    v = X.var(axis=0)
    X = X[:, v > 0]
    # z-score regions so high-signal regions don't dominate
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    pca = PCA(n_components=3, random_state=0)
    pc = pca.fit_transform(X)
    ve = pca.explained_variance_ratio_ * 100

    out = sub.copy()
    out['PC1'], out['PC2'], out['PC3'] = pc[:, 0], pc[:, 1], pc[:, 2]
    for i in range(3):
        out[f'PC{i+1}_var_pct'] = ve[i]
    out.to_csv(DATA_DIR / f'{fname}_coords.csv', index=False)

    # plot ALL pairwise PC combinations: (PC1,PC2), (PC1,PC3), (PC2,PC3).
    for (xi, yi) in [(1, 2), (1, 3), (2, 3)]:
        _plot_pc_pair(out, ve, xi, yi, f'{fname}_PC{xi}{yi}', title, panel_letter)
    print(f'  saved {fname} (PC1/2/3 pairwise): '
          f'PC1={ve[0]:.1f}% PC2={ve[1]:.1f}% PC3={ve[2]:.1f}% (n={keep.sum()})')


def _plot_pc_pair(out, ve, xi, yi, fname, title, panel_letter):
    xcol, ycol = f'PC{xi}', f'PC{yi}'
    # wider canvas: plot on the left, donor legend reserved on the right
    fig, ax = plt.subplots(figsize=(config.COL_SINGLE * 2.05, 5.0))
    # colour = stage, SHAPE = donor
    for _, r in out.iterrows():
        ax.scatter(r[xcol], r[ycol], s=140,
                   color=stage_color(r['stage']),
                   marker=DONOR_MARKERS.get(r['donor'], 'o'),
                   edgecolor='#333', linewidth=0.8, alpha=0.92, zorder=3)
    ax.set_xlabel('PC{} ({:.1f}%)'.format(xi, ve[xi - 1]))
    ax.set_ylabel('PC{} ({:.1f}%)'.format(yi, ve[yi - 1]))
    ax.set_title('{}  ·  PC{} vs PC{}'.format(title, xi, yi))

    def short(d):
        return d.replace('WTSI_OESO_', 'OESO')

    # Donor (shape) legend handles, ordered by stage.
    donor_order = (out[['donor', 'stage']].drop_duplicates()
                   .assign(_k=lambda d: d['stage'].map({s: i for i, s in enumerate(STAGE_ORDER)}))
                   .sort_values(['_k', 'donor']))
    donor_handles = [
        Line2D([0], [0], marker=DONOR_MARKERS.get(r.donor, 'o'), linestyle='',
               markerfacecolor=stage_color(r.stage), markeredgecolor='#333',
               markersize=11, label='{}  ({})'.format(short(r.donor), LABELS[r.stage]))
        for r in donor_order.itertuples()]
    # Donor (shape) legend BESIDE the plot (never over the data): shrink the
    # axes to ~68% width and anchor the legend in the reserved right margin.
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.68, box.height])
    leg_donor = ax.legend(handles=donor_handles, title='Donor (stage)',
                          loc='center left', bbox_to_anchor=(1.03, 0.5),
                          frameon=False, fontsize=10, title_fontsize=11,
                          handletextpad=0.4, labelspacing=0.55, borderpad=0.4)
    ax.add_artist(leg_donor)
    config.panel_label(ax, panel_letter)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.savefig(PNG_DIR / f'{fname}.png', dpi=300, bbox_inches='tight', facecolor='white')
    svg_path = SVG_DIR / f'{fname}.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    config.fix_svg_file(svg_path)
    plt.close(fig)


for tag, (sig_table, norm_label) in NORMS.items():
    if not (DATA_DIR / sig_table).exists():
        print(f'  [skip {tag}] {sig_table} not found')
        continue
    print(f'\n== normalisation: {norm_label} ==')
    mat, sheet = load_matrix(sig_table)
    run_pca_panel(mat, sheet, norm_label, PROGRESSION,
                  f'fig1_C_pca_progression_{tag}',
                  f'BG4 G4 signal PCA — progression ({norm_label})', 'C')
    run_pca_panel(mat, sheet, norm_label, STAGE_ORDER,
                  f'fig1_D_pca_with_Rx_{tag}',
                  f'BG4 G4 signal PCA — all samples ({norm_label})', 'D')
print('\nDone. PCA panels (nDR + CPM) in figures/, coords in data/.')
