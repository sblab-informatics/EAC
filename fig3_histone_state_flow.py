#!/usr/bin/env python3
"""
Figure 3 — Panel B: promoter chromatin-state flow across progression, split by G4.

Thick-stack alluvial of promoter chromatin STATE across NG -> ES -> LS, as two
side-by-side flows (G4+ | G4- promoters). State colours = histone legend.

Per-promoter marks (>=1 bp overlap with the CORRECTED reproducible, >=2-of-N
donor, manual-classification histone consensus): H3K27ac, H3K27me3, H3K4me1.

FOUR classification schemes are produced; each panel's title/legend states the
EXACT definition used:

  SCHEME 0  "ac_me3"   (no K4me1):
      Active=K27ac only · Bivalent=K27ac+K27me3 · Repressed=K27me3 only · Other=neither
  SCHEME A  "poised_primed" (K4me1, 6 states):
      Active=K27ac · Bivalent=K27ac+K27me3 · Poised=K4me1+K27me3(no K27ac)
      · Primed=K4me1 only · Repressed=K27me3 only · Other=none
  SCHEME B  "k4me1_into_bivalent" (K4me1 folded, 4 states; ~ old classify_state_v4):
      Active=K27ac(+/-K4me1) · Bivalent=(K27ac+K27me3) OR (K4me1+K27me3, no K27ac)
      · Repressed=K27me3 only · Other=none (incl. K4me1-only)
  SCHEME C  "single_primed" (K4me1, 5 states):
      Active=K27ac · Bivalent=K27ac+K27me3 · Primed=K4me1 without K27ac
      · Repressed=K27me3 only(no K4me1) · Other=none

x3 expression sets each: ALL · gene TPM>1 in any stage · gene TPM>0.1 in any stage.
Every panel writes its own *_rawdata.csv.
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
STATE_COLORS = {'Active': '#ff7f00', 'Bivalent': '#8B008B', 'Poised': '#4daf4a',
                'Primed': '#984ea3', 'Repressed': '#377eb8', 'Other': '#999999'}


# ---- classification schemes (ac, me3, me1 are bool) ------------------------
def cls_ac_me3(ac, me3, me1):
    if ac and not me3: return 'Active'
    if ac and me3: return 'Bivalent'
    if me3 and not ac: return 'Repressed'
    return 'Other'

def cls_poised_primed(ac, me3, me1):
    if ac and not me3: return 'Active'
    if ac and me3: return 'Bivalent'
    if me1 and me3 and not ac: return 'Poised'
    if me1 and not ac and not me3: return 'Primed'
    if me3 and not ac and not me1: return 'Repressed'
    return 'Other'

def cls_k4me1_into_bivalent(ac, me3, me1):
    if ac and not me3: return 'Active'
    if ac and me3: return 'Bivalent'
    if me3 and me1 and not ac: return 'Bivalent'
    if me3 and not ac and not me1: return 'Repressed'
    return 'Other'

def cls_single_primed(ac, me3, me1):
    if ac and not me3: return 'Active'
    if ac and me3: return 'Bivalent'
    if me1 and not ac: return 'Primed'
    if me3 and not ac and not me1: return 'Repressed'
    return 'Other'

SCHEMES = {
    'ac_me3':        (cls_ac_me3, ['Active', 'Bivalent', 'Repressed', 'Other'],
                      'Active=H3K27ac · Bivalent=H3K27ac+H3K27me3 · Repressed=H3K27me3 · Other=none'),
    'poised_primed': (cls_poised_primed, ['Active', 'Bivalent', 'Poised', 'Primed', 'Repressed', 'Other'],
                      'Active=H3K27ac · Bivalent=H3K27ac+H3K27me3 · Poised=H3K4me1+H3K27me3 · Primed=H3K4me1 · Repressed=H3K27me3 · Other=none'),
    'k4me1_bivalent': (cls_k4me1_into_bivalent, ['Active', 'Bivalent', 'Repressed', 'Other'],
                      'Active=H3K27ac · Bivalent=H3K27ac+H3K27me3 or H3K4me1+H3K27me3 · Repressed=H3K27me3 · Other=none'),
    'single_primed': (cls_single_primed, ['Active', 'Bivalent', 'Primed', 'Repressed', 'Other'],
                      'Active=H3K27ac · Bivalent=H3K27ac+H3K27me3 · Primed=H3K4me1(no H3K27ac) · Repressed=H3K27me3 · Other=none'),
}


def draw_alluvial(ax, df, tp_cols, states, colors, tp_labels,
                  count_fontsize=12, label_fontsize=18, min_h_label=0.05, min_h_full=0.085):
    n_tp = len(tp_cols); x_pos = np.arange(n_tp, dtype=float); gap = 0.008
    total = df.shape[0]
    if total == 0:
        ax.set_visible(False); return
    tp_counts = [df[c].value_counts().reindex(states, fill_value=0) for c in tp_cols]
    rect_positions = []
    for x, counts in zip(x_pos, tp_counts):
        positions = {}; y = 0.0
        for state in states:
            h = counts[state] / total
            positions[state] = (y, y + h)
            ax.add_patch(plt.Rectangle((x - 0.18, y), 0.36, h, facecolor=colors[state],
                                       edgecolor='white', linewidth=0.5, zorder=2))
            if h >= min_h_label:
                pct = 100.0 * counts[state] / total
                lab = ('{:,}\n({:.0f}%)'.format(counts[state], pct)
                       if h >= min_h_full else '{:.0f}%'.format(pct))
                ax.text(x, y + h / 2, lab, ha='center', va='center', fontsize=count_fontsize,
                        fontweight='bold', color='white' if state != 'Other' else '#333', zorder=3)
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
                h = c / total; y0, y1 = yf[sf], yt[st]
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
    ax.set_xticks(x_pos); ax.set_xticklabels(tp_labels, fontsize=label_fontsize)
    sns.despine(ax=ax, left=True); ax.set_yticks([])


LEGEND_LABEL = {
    'Active': 'Active (H3K27ac)', 'Bivalent': 'Bivalent (H3K27ac+H3K27me3)',
    'Poised': 'Poised (H3K4me1+H3K27me3)', 'Primed': 'Primed (H3K4me1)',
    'Repressed': 'Repressed (H3K27me3)', 'Other': 'Other (no mark)'}


def make_state_flow(df, states, fname, title, defn, groups, split_col):
    """N side-by-side chromatin-state alluvials, one per (label, mask) in `groups`.
    `split_col` is the per-promoter group column saved into the raw data."""
    sc = [f'state_{s}' for s in STAGES]
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(config.COL_FULL * (1.0 + 0.5 * n), 6.0),
                             gridspec_kw={'wspace': 0.35})
    if n == 1:
        axes = [axes]
    for ax, (label, sub) in zip(axes, groups):
        draw_alluvial(ax, sub, sc, states, STATE_COLORS, TP_LABELS)
        ax.set_xlabel('{} (n = {:,})'.format(label, len(sub)), fontsize=15, fontweight='bold')
    handles = [mpatches.Patch(color=STATE_COLORS[s], label=LEGEND_LABEL[s]) for s in states]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(states), 4), frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.05)
    # explicit state-definition line under the title
    fig.text(0.5, 1.005, defn, ha='center', va='bottom', fontsize=10.5, color='#555')
    config.panel_label(axes[0], 'B')
    fig.tight_layout()
    fig.savefig(PNG_DIR / f'{fname}.png', dpi=300, bbox_inches='tight', facecolor='white')
    svg = SVG_DIR / f'{fname}.svg'
    fig.savefig(svg, format='svg', bbox_inches='tight', facecolor='white')
    config.fix_svg_file(svg)
    plt.close(fig)
    cols = (['gene', 'ensg', 'chrom', 'start', 'end', 'strand', split_col]
            + sc + [f'{m}_{s}' for s in STAGES for m in ('H3K27ac', 'H3K27me3', 'H3K4me1')]
            + [f'tpm_{s}' for s in STAGES])
    cols = [c for c in cols if c in df.columns]
    df[cols].to_csv(DATA_DIR / f'{fname}_rawdata.csv', index=False)
    print(f'  {fname}: ' + ' / '.join('{} {:,}'.format(lab, len(sub)) for lab, sub in groups))


# ---- assemble per-promoter table -------------------------------------------
marks = pd.read_csv(DATA_DIR / 'promoter_histone_marks_corrected.csv')
g4 = pd.read_csv(DATA_DIR / 'promoter_g4_status_corrected.csv')[['gene', 'g4_NG', 'g4_ES_EAC', 'g4_LS_EAC']]
df = marks.merge(g4, on='gene', how='left')
df['g4_status_any'] = np.where((df[['g4_NG', 'g4_ES_EAC', 'g4_LS_EAC']] == 'G4+').any(axis=1), 'G4+', 'G4-')
# 3-way G4 trajectory (partitions all promoters; no overlap):
#   GAINED     = G4- at NG AND G4+ at ES or LS
#   G4+ at NG  = G4+ at NG (constitutive)
#   never G4+  = G4- in all 3 stages
df['g4_traj'] = np.select(
    [(df['g4_NG'] == 'G4-') & ((df['g4_ES_EAC'] == 'G4+') | (df['g4_LS_EAC'] == 'G4+')),
     df['g4_NG'] == 'G4+'],
    ['GAINED', 'G4+ at NG'], default='never G4+')

tpm = pd.read_csv(RNA_DIR / 'gene_tpm_matrix.csv')
meta = pd.read_csv(RNA_DIR / 'sample_metadata_corrected.csv')
gcol = 'gene_name' if 'gene_name' in tpm.columns else tpm.columns[0]
for st in STAGES:
    sids = [s for s in meta[meta['analytical_condition'] == st]['sample_id'] if s in tpm.columns]
    df[f'tpm_{st}'] = df['gene'].map(tpm.set_index(gcol)[sids].mean(axis=1))

tpm_cols = [f'tpm_{s}' for s in STAGES]
EXPR_SETS = {
    'all':       (df, 'all gencode v44 promoters'),
    'TPMgt1any': (df[(df[tpm_cols] > 1).any(axis=1)], 'gene TPM > 1 in any stage'),
    'TPMgt0.1any': (df[(df[tpm_cols] > 0.1).any(axis=1)], 'gene TPM > 0.1 in any stage'),
}

# ---- produce every scheme x every expression set ---------------------------
for sk, (clsf, states, defn) in SCHEMES.items():
    d2 = df.copy()
    for st in STAGES:
        d2[f'state_{st}'] = [clsf(bool(a), bool(m3), bool(m1)) for a, m3, m1 in
                             zip(d2[f'H3K27ac_{st}'], d2[f'H3K27me3_{st}'], d2[f'H3K4me1_{st}'])]
    for ek, (subdf_full, edesc) in EXPR_SETS.items():
        sub = d2.loc[subdf_full.index]

        # mode `any`: ever-G4+ vs never-G4+ (current 2-panel split)
        groups_any = [
            ('G4+ promoters', sub[sub['g4_status_any'] == 'G4+']),
            ('G4- promoters', sub[sub['g4_status_any'] == 'G4-']),
        ]
        make_state_flow(
            sub, states, f'fig3B_state_flow_{sk}_{ek}',
            'Chromatin-state flow [{}] · {} (n = {:,})'.format(sk, edesc, len(sub)),
            defn + '   |   split: G4+ = G4-positive in ANY stage (NG/ES/LS)',
            groups_any, 'g4_status_any')

        # mode `gained`: 3-way G4 trajectory (acquisition view)
        groups_gained = [
            ('GAINED (G4- NG -> G4+ ES/LS)', sub[sub['g4_traj'] == 'GAINED']),
            ('G4+ at NG (constitutive)', sub[sub['g4_traj'] == 'G4+ at NG']),
            ('never G4+', sub[sub['g4_traj'] == 'never G4+']),
        ]
        make_state_flow(
            sub, states, f'fig3B_gained_state_flow_{sk}_{ek}',
            'Chromatin-state flow by G4 trajectory [{}] · {} (n = {:,})'.format(sk, edesc, len(sub)),
            defn + '   |   GAINED = G4- at NG & G4+ at ES or LS',
            groups_gained, 'g4_traj')
print('done — all schemes x expression sets x {any, gained} splits')
