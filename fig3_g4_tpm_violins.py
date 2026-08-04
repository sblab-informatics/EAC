#!/usr/bin/env python3
"""
Figure 3 — Panel O: per-stage transcription of G4+ vs G4- promoters (violins).

For EACH stage (NG, ES_EAC, LS_EAC) one panel with TWO violins:
  G4+ promoters  vs  G4- promoters   (G4 status at THAT stage, g4_<stage>)
y = log2(TPM + 1) of the promoter's gene; expression measured at the SAME stage.

Genes are restricted to those EXPRESSED at the corresponding stage
(mean TPM > 0.1 over that stage's RNA-seq samples) — so each panel asks:
"among genes expressed at this stage, do G4+ promoters transcribe more than G4-?"

Statistics between the two groups, per stage:
  - two-sided Mann-Whitney U (rank test; raw + underflow-safe log10 p)
  - rank-biserial correlation = effect size (the primary read-out)
  - median log2(TPM+1) per group + Δmedian
  - BH-FDR across the 3 stages
Effect-size led; p bounded (1e-300 floor + log10 companion); unit = promoters
(genes), NOT donors (pseudo-replication caveat). Raw data + stats CSV saved.
"""
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, norm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA = SCRIPT_DIR / 'data'
RNA_DIR = DATA / 'rnaseq'
PNG = SCRIPT_DIR / 'figures' / 'png'
SVG = SCRIPT_DIR / 'figures' / 'svg'
config.setup_style()

MIN_P = 1e-300
TPM_THR = 0.1
STAGES = ['NG', 'ES_EAC', 'LS_EAC']
LAB = {'NG': 'NG', 'ES_EAC': 'ES', 'LS_EAC': 'LS'}
G4COL = {'NG': 'g4_NG', 'ES_EAC': 'g4_ES_EAC', 'LS_EAC': 'g4_LS_EAC'}
G4P, G4N = config.G4_COLORS['G4+'], config.G4_COLORS['G4-']   # red / blue
COL_HALF = config.COL_FULL * 0.48                              # single-panel width


def mw_log10p(x, y):
    """Mann-Whitney (two-sided) with asymptotic log10(p) so magnitude survives
    underflow, plus rank-biserial effect size. x vs y."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, np.nan
    U, p = mannwhitneyu(x, y, alternative='two-sided')
    n1, n2 = len(x), len(y)
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = abs((U - mu) / sd)
    lp = norm.logsf(z) / math.log(10) + math.log10(2)   # two-sided
    # rank-biserial signed so that >0 means the FIRST group (x) ranks HIGHER than y.
    # scipy's U counts wins of x over y, so rbc = 2U/(n1 n2) - 1 (intuitive sign).
    rbc = 2 * U / (n1 * n2) - 1
    return max(p, MIN_P), lp, rbc


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(1, n + 1)))[::-1])[::-1]
    return np.clip(q, 0, 1)


# ---- load promoters + per-stage TPM (same recipe as Panel N / state-flow) ----
g = pd.read_csv(DATA / 'promoter_g4_status_corrected.csv')
tpm = pd.read_csv(RNA_DIR / 'gene_tpm_matrix.csv')
meta = pd.read_csv(RNA_DIR / 'sample_metadata_corrected.csv')
gcol = 'gene_name' if 'gene_name' in tpm.columns else tpm.columns[0]
stage_tpm = {}
for st in STAGES:
    sids = [s for s in meta[meta['analytical_condition'] == st]['sample_id'] if s in tpm.columns]
    stage_tpm[st] = g['gene'].map(tpm.set_index(gcol)[sids].mean(axis=1))
    g[f'tpm_{st}'] = stage_tpm[st]


# ---- assemble long-form per-stage expressed table (vectorized) -------------
parts_long = []
for st in STAGES:
    expr = g[(g[f'tpm_{st}'] > TPM_THR) & (g[G4COL[st]].isin(['G4+', 'G4-']))]
    parts_long.append(pd.DataFrame({
        'stage': st, 'gene': expr['gene'].values, 'g4': expr[G4COL[st]].values,
        'tpm': expr[f'tpm_{st}'].values,
        'log2tpm': np.log2(expr[f'tpm_{st}'].values + 1)}))
long = pd.concat(parts_long, ignore_index=True)
long.to_csv(DATA / 'fig3O_g4_tpm_violins_rawdata.csv', index=False)


# ---- statistics per stage --------------------------------------------------
stats_rows = []
for st in STAGES:
    s = long[long.stage == st]
    xp = s[s.g4 == 'G4+'].log2tpm.values
    xn = s[s.g4 == 'G4-'].log2tpm.values
    p, lp, rbc = mw_log10p(xp, xn)
    stats_rows.append({
        'stage': st, 'test': 'mannwhitney_2sided', 'unit': 'promoters',
        'n_g4pos': int(len(xp)), 'n_g4neg': int(len(xn)),
        'median_log2tpm_g4pos': round(float(np.median(xp)), 4) if len(xp) else np.nan,
        'median_log2tpm_g4neg': round(float(np.median(xn)), 4) if len(xn) else np.nan,
        'delta_median_log2tpm': round(float(np.median(xp) - np.median(xn)), 4) if (len(xp) and len(xn)) else np.nan,
        'median_tpm_g4pos': round(float(np.median(s[s.g4 == 'G4+'].tpm)), 4) if len(xp) else np.nan,
        'median_tpm_g4neg': round(float(np.median(s[s.g4 == 'G4-'].tpm)), 4) if len(xn) else np.nan,
        'rank_biserial': round(float(rbc), 4),
        'p_raw': p, 'log10_p_raw': round(float(lp), 4),
    })
sdf = pd.DataFrame(stats_rows)
sdf['q_fdr'] = bh_fdr(sdf.p_raw.values).clip(min=MIN_P)
sdf['log10_q_fdr'] = np.log10(sdf.q_fdr)
sdf['p_raw'] = sdf.p_raw.clip(lower=MIN_P)
sdf.to_csv(DATA / 'fig3O_g4_tpm_violins_stats.csv', index=False)
qmap = dict(zip(sdf.stage, sdf.q_fdr))


# ============================ FIGURE ========================================
fig, axes = plt.subplots(1, 3, figsize=(config.COL_FULL * 1.5, 4.8),
                         gridspec_kw={'wspace': 0.30}, sharey=True)
ymax = long.log2tpm.max() * 1.18
for ai, st in enumerate(STAGES):
    ax = axes[ai]
    s = long[long.stage == st]
    data = [s[s.g4 == 'G4+'].log2tpm.values, s[s.g4 == 'G4-'].log2tpm.values]
    parts = ax.violinplot(data, positions=[0, 1], showextrema=False, widths=0.85)
    for pc, col in zip(parts['bodies'], [G4P, G4N]):
        pc.set_facecolor(col); pc.set_edgecolor(col); pc.set_alpha(0.55)
    # median + IQR box overlay
    for xi, (vals, col) in enumerate(zip(data, [G4P, G4N])):
        med = np.median(vals); q1, q3 = np.percentile(vals, [25, 75])
        ax.plot([xi, xi], [q1, q3], color='#222', lw=4, solid_capstyle='round', zorder=3)
        ax.plot(xi, med, 'o', color='white', mec='#222', ms=6, zorder=4)
        ax.text(xi, ymax * 0.99, 'med {:.2f}\nn={:,}'.format(med, len(vals)),
                ha='center', va='top', fontsize=8.5, color=col, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['G4+', 'G4−'], fontsize=12)
    ax.set_xlim(-0.65, 1.65); ax.set_ylim(0, ymax)
    # significance bracket between the two violins
    row = sdf[sdf.stage == st].iloc[0]
    star = config.stars_for_q(qmap[st]); star = star if star else 'ns'
    ybr = ymax * 0.86
    ax.plot([0, 0, 1, 1], [ybr, ybr + ymax * 0.015, ybr + ymax * 0.015, ybr],
            color='#222', lw=1.2)
    ax.text(0.5, ybr + ymax * 0.02,
            '{}\nrbc={:+.2f}'.format(star, row.rank_biserial),
            ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax.set_title('{}\nΔmed log2TPM = {:+.2f}'.format(LAB[st], row.delta_median_log2tpm),
                 fontsize=11.5, fontweight='bold')
    if ai == 0:
        ax.set_ylabel('log2(TPM + 1)')
    sns.despine(ax=ax)
config.panel_label(axes[0], 'O')
fig.suptitle('Transcription of G4+ vs G4− promoters per stage '
             '(genes with TPM > {} at that stage)\n'
             'two-sided Mann–Whitney; rbc = rank-biserial effect size; '
             '* q<0.05 ** q<0.01 *** q<0.001 **** q<1e-10 (BH-FDR across stages); unit = promoters'.format(TPM_THR),
             fontsize=11, fontweight='bold', y=1.02)
fig.savefig(PNG / 'fig3O_g4_tpm_violins.png', dpi=300, bbox_inches='tight', facecolor='white')
sv = SVG / 'fig3O_g4_tpm_violins.svg'
fig.savefig(sv, format='svg', bbox_inches='tight', facecolor='white')
config.fix_svg_file(sv)
plt.close(fig)


# ---- per-stage standalone panels (one file each) ---------------------------
for st in STAGES:
    s = long[long.stage == st]
    data = [s[s.g4 == 'G4+'].log2tpm.values, s[s.g4 == 'G4-'].log2tpm.values]
    figs, ax = plt.subplots(figsize=(COL_HALF, 4.4))
    parts = ax.violinplot(data, positions=[0, 1], showextrema=False, widths=0.85)
    for pc, col in zip(parts['bodies'], [G4P, G4N]):
        pc.set_facecolor(col); pc.set_edgecolor(col); pc.set_alpha(0.55)
    ymx = s.log2tpm.max() * 1.18
    for xi, (vals, col) in enumerate(zip(data, [G4P, G4N])):
        med = np.median(vals); q1, q3 = np.percentile(vals, [25, 75])
        ax.plot([xi, xi], [q1, q3], color='#222', lw=4, solid_capstyle='round', zorder=3)
        ax.plot(xi, med, 'o', color='white', mec='#222', ms=6, zorder=4)
        ax.text(xi, ymx * 0.99, 'med {:.2f}\nn={:,}'.format(med, len(vals)),
                ha='center', va='top', fontsize=8.5, color=col, fontweight='bold')
    row = sdf[sdf.stage == st].iloc[0]
    star = config.stars_for_q(qmap[st]); star = star if star else 'ns'
    ybr = ymx * 0.86
    ax.plot([0, 0, 1, 1], [ybr, ybr + ymx * 0.015, ybr + ymx * 0.015, ybr], color='#222', lw=1.2)
    ax.text(0.5, ybr + ymx * 0.02, '{}  rbc={:+.2f}'.format(star, row.rank_biserial),
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['G4+', 'G4−'], fontsize=12)
    ax.set_xlim(-0.65, 1.65); ax.set_ylim(0, ymx)
    ax.set_ylabel('log2(TPM + 1)')
    ax.set_title('{} · G4+ vs G4− promoter transcription\n'
                 '(TPM>{} at {}); Δmed={:+.2f}, q={:.1e}'.format(
                     LAB[st], TPM_THR, LAB[st], row.delta_median_log2tpm, qmap[st]),
                 fontsize=10.5, fontweight='bold')
    config.panel_label(ax, 'O')
    sns.despine(ax=ax)
    figs.savefig(PNG / f'fig3O_g4_tpm_violin_{st}.png', dpi=300, bbox_inches='tight', facecolor='white')
    svf = SVG / f'fig3O_g4_tpm_violin_{st}.svg'
    figs.savefig(svf, format='svg', bbox_inches='tight', facecolor='white')
    config.fix_svg_file(svf)
    plt.close(figs)

print('Panel O — G4+/G4− promoter transcription per stage:')
print(sdf[['stage', 'n_g4pos', 'n_g4neg', 'median_log2tpm_g4pos', 'median_log2tpm_g4neg',
           'delta_median_log2tpm', 'rank_biserial', 'log10_p_raw', 'q_fdr']].to_string(index=False))
