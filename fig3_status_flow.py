#!/usr/bin/env python3
"""
Figure 3 — Panel L: promoter G4-status flow across progression (NG -> ES -> LS).

Thick-stack alluvial of per-promoter G4 status (G4+ red / G4- blue) across the
three progression stages. Style = the 11May fig3L template (wide stacks with
count + % readable inside; ribbons at low alpha). Two versions:

  fig3L_g4_status_flow          — ALL promoters (gencode v44, 38,841)
  fig3L2_g4_status_flow_active  — ACTIVE promoters only (gene TPM > 0.1)

G4 status from the CORRECTED reproducible (>=2-of-N donor, manual classification)
BG4 sets: data/promoter_g4_status_corrected.csv. Activity from the CORRECTED
RNA-seq (analytical_condition; tech reps averaged to stage mean TPM).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MPath
import numpy as np
import pandas as pd
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA_DIR = SCRIPT_DIR / 'data'
PNG_DIR = SCRIPT_DIR / 'figures' / 'png'
SVG_DIR = SCRIPT_DIR / 'figures' / 'svg'
RNA_DIR = SCRIPT_DIR / 'data' / 'rnaseq'
for d in (PNG_DIR, SVG_DIR):
    d.mkdir(parents=True, exist_ok=True)
config.setup_style()

STAGES = ['NG', 'ES_EAC', 'LS_EAC']
TP_LABELS = ['NG', 'ES', 'LS']
G4_STATES = ['G4+', 'G4-']
G4_COLORS = {'G4+': config.G4_COLORS['G4+'], 'G4-': config.G4_COLORS['G4-']}  # red / blue
TPM_ACTIVE = 0.1   # active promoter threshold


# ---- thick-stack alluvial (vendored from 11May fig3 template) --------------
def draw_alluvial(ax, df, tp_cols, states, colors, tp_labels=None,
                  count_fontsize=17, label_fontsize=21,
                  min_rect_height_for_label=0.06,
                  min_rect_height_for_full_label=0.09):
    n_tp = len(tp_cols)
    x_pos = np.arange(n_tp, dtype=float)
    gap = 0.01
    total = df.shape[0]
    if total == 0:
        ax.set_visible(False); return
    tp_counts = [df[c].value_counts().reindex(states, fill_value=0) for c in tp_cols]

    rect_positions = []
    for x, counts in zip(x_pos, tp_counts):
        positions = {}
        y = 0.0
        for state in states:
            h = counts[state] / total
            positions[state] = (y, y + h)
            ax.add_patch(plt.Rectangle((x - 0.18, y), 0.36, h, facecolor=colors[state],
                                       edgecolor='white', linewidth=0.5, zorder=2))
            if h >= min_rect_height_for_label:
                pct = 100.0 * counts[state] / total
                lab = ('{:,}\n({:.1f}%)'.format(counts[state], pct)
                       if h >= min_rect_height_for_full_label else '{:.1f}%'.format(pct))
                ax.text(x, y + h / 2, lab, ha='center', va='center',
                        fontsize=count_fontsize, fontweight='bold',
                        color='white', zorder=3)
            y += h + gap
        rect_positions.append(positions)

    for ti in range(n_tp - 1):
        cf, ct = tp_cols[ti], tp_cols[ti + 1]
        xf, xt = x_pos[ti], x_pos[ti + 1]
        trans = pd.crosstab(df[cf], df[ct]).reindex(index=states, columns=states, fill_value=0)
        yf = {s: rect_positions[ti][s][0] for s in states}
        yt = {s: rect_positions[ti + 1][s][0] for s in states}
        for sf in states:
            for st in states:
                c = trans.loc[sf, st]
                if c == 0:
                    continue
                h = c / total
                y0, y1 = yf[sf], yt[st]
                verts = [(xf + 0.18, y0), ((xf + xt) / 2, y0), ((xf + xt) / 2, y1),
                         (xt - 0.18, y1), (xt - 0.18, y1 + h), ((xf + xt) / 2, y1 + h),
                         ((xf + xt) / 2, y0 + h), (xf + 0.18, y0 + h), (xf + 0.18, y0)]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4,
                         MPath.LINETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
                ax.add_patch(mpatches.PathPatch(MPath(verts, codes), facecolor=colors[sf],
                                                alpha=0.25, edgecolor='none', zorder=1))
                yf[sf] += h; yt[st] += h

    ax.set_xlim(-0.5, n_tp - 0.5)
    ax.set_ylim(-0.02, 1.0 + gap * (len(states) - 1) + 0.05)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(tp_labels, fontsize=label_fontsize)
    ax.set_ylabel('Fraction of promoters', fontsize=label_fontsize)
    sns.despine(ax=ax, left=True)
    ax.set_yticks([])
    handles = [mpatches.Patch(color=colors[s], label=s) for s in states]
    ax.legend(handles=handles, loc='upper right', fontsize=18, frameon=False,
              bbox_to_anchor=(1.25, 1.0))


def transition_table(df, subset_name):
    cols = [f'g4_{s}' for s in STAGES]
    rows = []
    for step, a, b in [('NG->ES', cols[0], cols[1]), ('ES->LS', cols[1], cols[2]),
                       ('NG->LS', cols[0], cols[2])]:
        gained = int(((df[a] == 'G4-') & (df[b] == 'G4+')).sum())
        lost = int(((df[a] == 'G4+') & (df[b] == 'G4-')).sum())
        sp = int(((df[a] == 'G4+') & (df[b] == 'G4+')).sum())
        sn = int(((df[a] == 'G4-') & (df[b] == 'G4-')).sum())
        rows.append({'subset_name': subset_name, 'transition': step,
                     'gained_g4': gained, 'lost_g4': lost,
                     'stable_g4pos': sp, 'stable_g4neg': sn, 'net_gain': gained - lost})
    return pd.DataFrame(rows)


def make_flow(df, fname, panel_letter, title):
    """title = explicit description of the promoter set + n (printed on the panel)."""
    cols = [f'g4_{s}' for s in STAGES]
    fig, ax = plt.subplots(figsize=(config.COL_FULL, 5.8))
    draw_alluvial(ax, df, tp_cols=cols, tp_labels=TP_LABELS,
                  states=G4_STATES, colors=G4_COLORS)
    # +gained / -lost between consecutive stages, mid-height
    for ti, (a, b) in enumerate([(cols[0], cols[1]), (cols[1], cols[2])]):
        gained = int(((df[a] == 'G4-') & (df[b] == 'G4+')).sum())
        lost = int(((df[a] == 'G4+') & (df[b] == 'G4-')).sum())
        ax.text(ti + 0.5, 0.53, '+{:,}'.format(gained), ha='center', va='bottom',
                fontsize=15, fontweight='bold', color=G4_COLORS['G4+'],
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1), zorder=5)
        ax.text(ti + 0.5, 0.47, '−{:,}'.format(lost), ha='center', va='top',
                fontsize=14, color=G4_COLORS['G4-'],
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1), zorder=5)
    ax.set_title(title, fontsize=15, fontweight='bold')
    config.panel_label(ax, panel_letter)
    fig.tight_layout()
    fig.savefig(PNG_DIR / f'{fname}.png', dpi=300, bbox_inches='tight', facecolor='white')
    svg = SVG_DIR / f'{fname}.svg'
    fig.savefig(svg, format='svg', bbox_inches='tight', facecolor='white')
    with open(svg) as fh:
        t = fh.read()
    with open(svg, 'w') as fh:
        fh.write(config.fix_svg_for_affinity(t))
    plt.close(fig)
    # RAW DATA for this exact panel: every promoter, its G4 status at each stage,
    # plus the gene's per-stage mean TPM and the active flag.
    raw = df[['gene', 'ensg', 'chrom', 'start', 'end', 'strand',
              'g4_NG', 'g4_ES_EAC', 'g4_LS_EAC']].copy()
    raw['tpm_NG'] = raw['gene'].map(stage_mean['NG'])
    raw['tpm_ES_EAC'] = raw['gene'].map(stage_mean['ES_EAC'])
    raw['tpm_LS_EAC'] = raw['gene'].map(stage_mean['LS_EAC'])
    raw['active_TPMgt0.1_all3'] = raw['gene'].isin(active_genes)
    raw['active_TPMgt0.1_any'] = raw['gene'].isin(active_any_genes)
    raw.to_csv(DATA_DIR / f'{fname}_rawdata.csv', index=False)
    print(f'  saved {fname} (n={len(df):,}) + {fname}_rawdata.csv')


# ---- data ------------------------------------------------------------------
prom = pd.read_csv(DATA_DIR / 'promoter_g4_status_corrected.csv')

# active gene set: TPM > 0.1 in ALL 3 progression stages (corrected RNA-seq)
tpm = pd.read_csv(RNA_DIR / 'gene_tpm_matrix.csv')
meta = pd.read_csv(RNA_DIR / 'sample_metadata_corrected.csv')
gcol = 'gene_name' if 'gene_name' in tpm.columns else tpm.columns[0]
stage_mean = {}
for st in STAGES:
    sids = [s for s in meta[meta['analytical_condition'] == st]['sample_id'] if s in tpm.columns]
    stage_mean[st] = tpm.set_index(gcol)[sids].mean(axis=1)
mask_all3 = (stage_mean['NG'] > TPM_ACTIVE) & (stage_mean['ES_EAC'] > TPM_ACTIVE) & (stage_mean['LS_EAC'] > TPM_ACTIVE)
mask_any = (stage_mean['NG'] > TPM_ACTIVE) | (stage_mean['ES_EAC'] > TPM_ACTIVE) | (stage_mean['LS_EAC'] > TPM_ACTIVE)
active_genes = set(stage_mean['NG'][mask_all3].index)        # "all 3" set (for raw-data flag)
active_any_genes = set(stage_mean['NG'][mask_any].index)     # "any stage" set
prom_active = prom[prom['gene'].isin(active_genes)].copy()
prom_active_any = prom[prom['gene'].isin(active_any_genes)].copy()
print(f"all promoters: {len(prom):,} | active all-3: {len(prom_active):,} | active any-stage: {len(prom_active_any):,}")

# ---- panels (title states the exact set + n) -------------------------------
make_flow(prom, 'fig3L_g4_status_flow', 'L',
          title='All gencode v44 promoters (n = {:,})'.format(len(prom)))
make_flow(prom_active, 'fig3L2_g4_status_flow_active', 'L',
          title='Active promoters · gene TPM > {} in all 3 stages (n = {:,})'.format(
              TPM_ACTIVE, len(prom_active)))
make_flow(prom_active_any, 'fig3L3_g4_status_flow_active_any', 'L',
          title='Active promoters · gene TPM > {} in any stage (n = {:,})'.format(
              TPM_ACTIVE, len(prom_active_any)))

# ---- data tables -----------------------------------------------------------
transition_table(prom, 'ALL_PROMOTERS').to_csv(DATA_DIR / 'fig3L_g4_transitions.csv', index=False)
transition_table(prom_active, 'ACTIVE_TPMgt0.1_all3').to_csv(DATA_DIR / 'fig3L2_g4_transitions_active.csv', index=False)
transition_table(prom_active_any, 'ACTIVE_TPMgt0.1_any').to_csv(DATA_DIR / 'fig3L3_g4_transitions_active_any.csv', index=False)
print('saved transition tables')
