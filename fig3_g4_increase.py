#!/usr/bin/env python3
"""
Figure 3 — Panel A2: G4+ promoter increase per chromatin state across progression.

For each chromatin STATE, the number and the % of promoters in that state that
are G4+, plotted across NG -> ES -> LS (one line per state, state colours).
Re-make of 11May fig3A2_g4_increase with our corrected data.

  state   = classify_*(K27ac, K27me3, K4me1) from the corrected stringent-union
            histone sets (per-stage)
  G4+/-   = corrected reproducible (>=2-of-N donor) BG4 sets
  pct_g4p = 100 * (#G4+ promoters in state) / (#promoters in state)   [per stage]

Built for all 4 state schemes (ac_me3, poised_primed, k4me1_bivalent,
single_primed); each labelled. Stats: effect size led (G4+ %-point change
NG->LS per state), Fisher NG-vs-LS p reported bounded + flagged n=promoters
(NOT donors). Three output forms per scheme: _counts, _pct, _combined.
Every panel writes its own *_rawdata.csv.
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
from scipy.stats import fisher_exact

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import config  # noqa: E402

DATA_DIR = SCRIPT_DIR / 'data'
PNG_DIR = SCRIPT_DIR / 'figures' / 'png'
SVG_DIR = SCRIPT_DIR / 'figures' / 'svg'
for d in (PNG_DIR, SVG_DIR):
    d.mkdir(parents=True, exist_ok=True)
config.setup_style()

STAGES = ['NG', 'ES_EAC', 'LS_EAC']
X = [0, 1, 2]
X_LAB = ['NG', 'ES', 'LS']
STATE_COLORS = {'Active': '#ff7f00', 'Bivalent': '#8B008B', 'Poised': '#4daf4a',
                'Primed': '#984ea3', 'Repressed': '#377eb8', 'Other': '#999999'}
# Always describe each state by its histone modification(s) in the legend.
STATE_HISTONE_LABEL = {
    'Active':    'Active (H3K27ac)',
    'Bivalent':  'Bivalent (H3K27ac+H3K27me3)',
    'Poised':    'Poised (H3K4me1+H3K27me3)',
    'Primed':    'Primed (H3K4me1)',
    'Repressed': 'Repressed (H3K27me3)',
    'Other':     'Other (no mark)',
}
MIN_P = 1e-300


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

def cls_k4me1_bivalent(ac, me3, me1):
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
    'ac_me3': (cls_ac_me3, ['Active', 'Bivalent', 'Repressed', 'Other']),
    'poised_primed': (cls_poised_primed, ['Active', 'Bivalent', 'Poised', 'Primed', 'Repressed', 'Other']),
    'k4me1_bivalent': (cls_k4me1_bivalent, ['Active', 'Bivalent', 'Repressed', 'Other']),
    'single_primed': (cls_single_primed, ['Active', 'Bivalent', 'Primed', 'Repressed', 'Other']),
}


def fmt_p_bounded(p):
    if not np.isfinite(p) or p <= MIN_P:
        return 'p<1e-300'
    return 'p={:.1e}'.format(p)


# ---- data ------------------------------------------------------------------
g4 = pd.read_csv(DATA_DIR / 'promoter_g4_status_corrected.csv')
marks = pd.read_csv(DATA_DIR / 'promoter_histone_marks_corrected.csv')
df0 = marks.merge(g4[['gene'] + [f'g4_{s}' for s in STAGES]], on='gene')


def build_table(scheme):
    clsf, states = SCHEMES[scheme]
    d = df0.copy()
    for s in STAGES:
        d[f'state_{s}'] = [clsf(bool(a), bool(m), bool(k)) for a, m, k in
                           zip(d[f'H3K27ac_{s}'], d[f'H3K27me3_{s}'], d[f'H3K4me1_{s}'])]
    rows = []
    for s in STAGES:
        for st in states:
            in_state = d[f'state_{s}'] == st
            n_state = int(in_state.sum())
            n_g4p = int((in_state & (d[f'g4_{s}'] == 'G4+')).sum())
            rows.append({'state': st, 'stage': s, 'n_state': n_state, 'n_g4p': n_g4p,
                         'pct_g4p': round(100 * n_g4p / n_state, 1) if n_state else 0.0})
    return d, states, pd.DataFrame(rows)


def stats_table(tab, states):
    rows = []
    for st in states:
        ng = tab[(tab.state == st) & (tab.stage == 'NG')].iloc[0]
        ls = tab[(tab.state == st) & (tab.stage == 'LS_EAC')].iloc[0]
        table = [[ng.n_g4p, ng.n_state - ng.n_g4p], [ls.n_g4p, ls.n_state - ls.n_g4p]]
        try:
            orr, p = fisher_exact(table)
        except ValueError:
            orr, p = np.nan, np.nan
        rows.append({
            'state': st, 'comparison': 'NG_vs_LS', 'unit': 'promoters (NOT donors)',
            'pct_g4p_NG': ng.pct_g4p, 'pct_g4p_LS': ls.pct_g4p,
            'effect_pct_point_change': round(ls.pct_g4p - ng.pct_g4p, 1),
            'odds_ratio': orr,
            'p_raw': max(p, MIN_P) if np.isfinite(p) else np.nan,
            'log10_p_raw': float(math.log10(max(p, MIN_P))) if np.isfinite(p) and p > 0 else
                           (math.log10(MIN_P) if np.isfinite(p) else np.nan),
        })
    return pd.DataFrame(rows)


def plot_panel(tab, states, fname, kind, scheme):
    """kind: 'counts' | 'pct' | 'combined'."""
    if kind == 'combined':
        fig, axes = plt.subplots(1, 2, figsize=(config.COL_FULL, 4.8),
                                 gridspec_kw={'wspace': 0.4})
        _line(axes[0], tab, states, 'counts'); _line(axes[1], tab, states, 'pct')
        config.panel_label(axes[0], 'A')
    else:
        fig, ax = plt.subplots(figsize=(config.COL_1_5, 4.8))
        _line(ax, tab, states, kind)
        config.panel_label(ax, 'A')
    fig.suptitle('G4+ promoters per chromatin state [{}]'.format(scheme),
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(PNG_DIR / f'{fname}.png', dpi=300, bbox_inches='tight', facecolor='white')
    svg = SVG_DIR / f'{fname}.svg'
    fig.savefig(svg, format='svg', bbox_inches='tight', facecolor='white')
    with open(svg) as fh:
        t = fh.read()
    with open(svg, 'w') as fh:
        fh.write(config.fix_svg_for_affinity(t))
    plt.close(fig)


def _line(ax, tab, states, kind):
    ycol = 'n_g4p' if kind == 'counts' else 'pct_g4p'
    for st in states:
        sd = tab[tab.state == st].set_index('stage').loc[STAGES]
        y = sd[ycol].values
        ax.plot(X, y, '-o', color=STATE_COLORS[st], linewidth=2.4, markersize=9,
                markeredgecolor='white', markeredgewidth=0.8,
                label=STATE_HISTONE_LABEL[st], zorder=3)
        for xi, v in zip(X, y):
            lab = '{:,}'.format(int(v)) if kind == 'counts' else '{:.0f}%'.format(v)
            ax.annotate(lab, xy=(xi, v), xytext=(0, 9), textcoords='offset points',
                        fontsize=13, ha='center', va='bottom', color=STATE_COLORS[st],
                        fontweight='bold')
    ax.set_xticks(X); ax.set_xticklabels(X_LAB)
    if kind == 'counts':
        ax.set_ylabel('# G4+ promoters')
    else:
        ax.set_ylabel('% G4+ within state'); ax.set_ylim(-2, 110)
    ax.set_xlim(-0.3, 2.3)
    ax.legend(frameon=False, loc='upper left', fontsize=11)
    sns.despine(ax=ax)


for scheme in SCHEMES:
    d, states, tab = build_table(scheme)
    tab.to_csv(DATA_DIR / f'fig3A2_g4_increase_{scheme}_rawdata.csv', index=False)
    stats_table(tab, states).to_csv(DATA_DIR / f'fig3A2_g4_increase_{scheme}_stats.csv', index=False)
    for kind in ('combined', 'counts', 'pct'):
        plot_panel(tab, states, f'fig3A2_g4_increase_{scheme}_{kind}', kind, scheme)
    # report
    act = tab[(tab.state == 'Active')].set_index('stage')['pct_g4p']
    print(f"  {scheme}: Active G4+% NG={act.get('NG',float('nan'))} -> LS={act.get('LS_EAC',float('nan'))}")
print('done — fig3A2 for all schemes (counts/pct/combined) + stats + rawdata')
