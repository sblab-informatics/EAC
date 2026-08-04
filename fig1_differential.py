#!/usr/bin/env python3
"""
Figure 1 — Panel E/F: differential BG4 G4 signal between stages.

Per-region donor-level differential over the G4 universe. For each universe
region: technical replicates are averaged to a DONOR mean (the biological unit),
then stage contrasts are computed across donors.

  effect size = log2 fold-change of mean donor signal (stageB / stageA)
  test        = Mann-Whitney U across donors (per region) WHERE BOTH stages
                have >=2 donors; FDR (BH) across regions. Contrasts touching a
                single-donor stage (NG, ES_Rx, LS_Rx) cannot be tested and are
                reported as log2FC effect-size ONLY (descriptive, flagged n=1) —
                no p-value, per the donor-as-unit methodology.

Run for BOTH normalisations (nDR Drosophila spike-in, and CPM) so differential
regions can be compared between normalisation schemes.

Output: per-contrast differential tables (data/) + MA / volcano panels (figs).
Effect size leads; p (where computable) is FDR-corrected and bounded.
"""
import sys
import itertools
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA_DIR = SCRIPT_DIR / 'data'
PNG_DIR = SCRIPT_DIR / 'figures' / 'png'
SVG_DIR = SCRIPT_DIR / 'figures' / 'svg'
for d in (DATA_DIR, PNG_DIR, SVG_DIR):
    d.mkdir(parents=True, exist_ok=True)
config.setup_style()

NORMS = {
    'nDR': ('fig1_bg4_signal_over_universe.tab', 'Drosophila spike-in (nDR)'),
    'cpm': ('fig1_bg4_signal_over_universe_cpm.tab', 'CPM'),
}
STAGE_ORDER = ['NG', 'ES_EAC', 'LS_EAC', 'ES_Rx', 'LS_Rx']
LABELS = {'NG': 'NG', 'ES_EAC': 'ES', 'LS_EAC': 'LS', 'ES_Rx': 'ES+Rx', 'LS_Rx': 'LS+Rx'}
SHEET = pd.read_csv(DATA_DIR / 'fig1_sample_sheet.csv')
SHEET = SHEET[SHEET['stage'].isin(STAGE_ORDER)].reset_index(drop=True)  # drop NDBE
LFC_THR = 1.0   # |log2FC| >= 1 (2-fold)
FDR_THR = 0.05


def bh_fdr(p):
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    q = np.full_like(p, np.nan)
    pv = p[ok]
    n = len(pv)
    order = np.argsort(pv)
    ranked = pv[order]
    qv = ranked * n / (np.arange(1, n + 1))
    qv = np.minimum.accumulate(qv[::-1])[::-1]
    qq = np.empty(n)
    qq[order] = np.clip(qv, 0, 1)
    q[ok] = qq
    return q


def load_donor_matrix(sig_table):
    """Return regions DataFrame and a donor-mean signal matrix (regions x donors)
    plus donor->stage map. Tech reps averaged per donor; log2(signal+1)."""
    sig = pd.read_csv(DATA_DIR / sig_table, sep='\t')
    sig.columns = [c.strip("#'\" ") for c in sig.columns]
    regions = sig[['chr', 'start', 'end']].copy()
    sample_cols = [c for c in sig.columns if c not in ('chr', 'start', 'end')]
    # map sample column -> (donor, stage)
    lab = {f"{r.sample_id}_{r.donor}_{r.stage}_{r.trep}": (r.donor, r.stage)
           for r in SHEET.itertuples()}
    valid = [c for c in sample_cols if c in lab]
    X = np.log2(sig[valid].fillna(0.0).values + 1.0)   # regions x samples
    donors = sorted({lab[c][0] for c in valid})
    donor_stage = {lab[c][0]: lab[c][1] for c in valid}
    # donor mean across that donor's tech reps
    D = np.zeros((X.shape[0], len(donors)))
    for j, d in enumerate(donors):
        cols = [i for i, c in enumerate(valid) if lab[c][0] == d]
        D[:, j] = X[:, cols].mean(axis=1)
    return regions, D, donors, donor_stage


def differential(regions, D, donors, donor_stage, sA, sB):
    """log2FC (B - A in log2 space) per region + donor-level MWU where testable."""
    iA = [j for j, d in enumerate(donors) if donor_stage[d] == sA]
    iB = [j for j, d in enumerate(donors) if donor_stage[d] == sB]
    nA, nB = len(iA), len(iB)
    meanA = D[:, iA].mean(axis=1)
    meanB = D[:, iB].mean(axis=1)
    lfc = meanB - meanA                      # already log2 space
    testable = (nA >= 2 and nB >= 2)
    pvals = np.full(D.shape[0], np.nan)
    if testable:
        a = D[:, iA]; b = D[:, iB]
        for r in range(D.shape[0]):
            try:
                _, p = mannwhitneyu(a[r], b[r], alternative='two-sided')
            except ValueError:
                p = np.nan
            pvals[r] = p
    q = bh_fdr(pvals) if testable else np.full(D.shape[0], np.nan)
    out = regions.copy()
    out['mean_A_log2'] = meanA
    out['mean_B_log2'] = meanB
    out['log2FC'] = lfc
    out['p_raw'] = pvals
    out['q_fdr'] = q
    return out, nA, nB, testable


def ma_panel(df, sA, sB, nA, nB, testable, norm_label, fname, panel_letter):
    fig, ax = plt.subplots(figsize=(config.COL_SINGLE * 1.5, 4.6))
    A = (df['mean_A_log2'] + df['mean_B_log2']) / 2   # mean signal (x)
    y = df['log2FC']
    if testable:
        sig = (df['q_fdr'] < FDR_THR) & (df['log2FC'].abs() >= LFC_THR)
        ax.scatter(A[~sig], y[~sig], s=6, color='#bbbbbb', alpha=0.5, zorder=2, rasterized=True)
        up = sig & (y > 0); dn = sig & (y < 0)
        ax.scatter(A[up], y[up], s=8, color=config.CONDITION_COLORS.get(sB, '#B2182B'),
                   alpha=0.8, zorder=3, rasterized=True)
        ax.scatter(A[dn], y[dn], s=8, color=config.CONDITION_COLORS.get(sA, '#2166AC'),
                   alpha=0.8, zorder=3, rasterized=True)
        ntag = ('up={} ({}) / down={} ({})\nFDR<{}, |log2FC|>={}'
                .format(int(up.sum()), LABELS[sB], int(dn.sum()), LABELS[sA], FDR_THR, LFC_THR))
    else:
        ax.scatter(A, y, s=6, color='#bbbbbb', alpha=0.5, zorder=2, rasterized=True)
        big = y.abs() >= LFC_THR
        ax.scatter(A[big & (y > 0)], y[big & (y > 0)], s=8,
                   color=config.CONDITION_COLORS.get(sB, '#B2182B'), alpha=0.8, zorder=3, rasterized=True)
        ax.scatter(A[big & (y < 0)], y[big & (y < 0)], s=8,
                   color=config.CONDITION_COLORS.get(sA, '#2166AC'), alpha=0.8, zorder=3, rasterized=True)
        ntag = ('|log2FC|>={}: up={} / down={}\nEFFECT SIZE ONLY (n=1 stage, no test)'
                .format(LFC_THR, int((big & (y > 0)).sum()), int((big & (y < 0)).sum())))
    ax.axhline(0, color='#333', lw=0.8)
    ax.axhline(LFC_THR, color='#888', lw=0.6, ls='--')
    ax.axhline(-LFC_THR, color='#888', lw=0.6, ls='--')
    ax.set_xlabel('mean log2 signal')
    ax.set_ylabel('log2 FC  ({} → {})'.format(LABELS[sA], LABELS[sB]))
    ax.set_title('{} vs {}  ({})'.format(LABELS[sA], LABELS[sB], norm_label), fontsize=14)
    ax.text(0.02, 0.97, ntag + '\nn={} vs {} donors'.format(nA, nB),
            transform=ax.transAxes, va='top', ha='left', fontsize=11,
            bbox=dict(facecolor='white', edgecolor='#ccc', alpha=0.85, pad=3))
    config.panel_label(ax, panel_letter)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(PNG_DIR / f'{fname}.png', dpi=300, bbox_inches='tight', facecolor='white')
    svg_path = SVG_DIR / f'{fname}.svg'
    fig.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    with open(svg_path) as fh:
        txt = fh.read()
    with open(svg_path, 'w') as fh:
        fh.write(config.fix_svg_for_affinity(txt))
    plt.close(fig)


CONTRASTS = list(itertools.combinations(STAGE_ORDER, 2))  # all pairwise

summary_rows = []
for tag, (sig_table, norm_label) in NORMS.items():
    if not (DATA_DIR / sig_table).exists():
        print(f'[skip {tag}] {sig_table} missing'); continue
    print(f'\n== {norm_label} ==')
    regions, D, donors, donor_stage = load_donor_matrix(sig_table)
    # Save the per-DONOR-mean signal matrix per region (tech-reps collapsed),
    # log2(signal+1). This is the rawest donor-level table: every region x every
    # donor, so any contrast can be re-derived at the biological (donor) unit.
    donor_mat = regions.copy()
    for j, d in enumerate(donors):
        donor_mat['{}__{}'.format(d, donor_stage[d])] = D[:, j]
    donor_mat.to_csv(DATA_DIR / f'fig1_donor_mean_signal_per_region_{tag}.csv', index=False)
    print(f'  saved per-donor-mean signal matrix ({len(donors)} donors x {len(regions)} regions)')
    for sA, sB in CONTRASTS:
        df, nA, nB, testable = differential(regions, D, donors, donor_stage, sA, sB)
        cname = f'{sA}_vs_{sB}'
        df.to_csv(DATA_DIR / f'fig1_diff_{cname}_{tag}.csv', index=False)
        pl = 'E' if (sA, sB) == ('ES_EAC', 'LS_EAC') else ('F' if (sA, sB) == ('NG', 'LS_EAC') else '')
        fname = f'fig1_diff_{cname}_{tag}'
        ma_panel(df, sA, sB, nA, nB, testable, norm_label, fname, pl or 'E')
        if testable:
            sig = (df['q_fdr'] < FDR_THR) & (df['log2FC'].abs() >= LFC_THR)
            n_up = int((sig & (df['log2FC'] > 0)).sum())
            n_dn = int((sig & (df['log2FC'] < 0)).sum())
        else:
            big = df['log2FC'].abs() >= LFC_THR
            n_up = int((big & (df['log2FC'] > 0)).sum())
            n_dn = int((big & (df['log2FC'] < 0)).sum())
        summary_rows.append({
            'norm': tag, 'contrast': cname, 'n_donors_A': nA, 'n_donors_B': nB,
            'testable_donor_level': testable, 'n_regions': len(df),
            'n_up_in_B': n_up, 'n_down_in_B': n_dn,
            'median_log2FC': float(df['log2FC'].median()),
        })
        print(f'  {cname:18s} n={nA}v{nB} testable={testable} up={n_up} down={n_dn}')

summ = pd.DataFrame(summary_rows)
# add log10 companions to the per-contrast CSVs (schema) via the shared retrofit
summ.to_csv(DATA_DIR / 'fig1_differential_summary.csv', index=False)
print('\nSummary -> data/fig1_differential_summary.csv')
print(summ.to_string(index=False))
