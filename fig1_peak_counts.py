#!/usr/bin/env python3
"""
Figure 1 — Panel A/B: BG4 G4 peak counts per stage (manual classification).

Nature-style presentation: per-DONOR scatter (each donor = one point) with the
stage MEDIAN and error (IQR), coloured by condition (config.CONDITION_COLORS).
No bar charts. Single-donor stages (NG, ES_Rx, LS_Rx) show the lone point and
are flagged n=1 (no error, hatched marker edge).

Each donor's peak count is its own >=2-of-3 technical-replicate SEACR BG4
consensus; donors are grouped into stages by the MANUAL classification
(CAM450->LS_EAC, WTSI_OESO_117->ES_EAC).

Two versions:
  Panel A  — progression only (NG, ES_EAC, LS_EAC)
  Panel B  — including Rx (NG, ES_EAC, LS_EAC, ES_Rx, LS_Rx)

Data source: Rebecca_pre_manuscript SEACR_consensus/BG4/patient_level/
             consensus_<donor>.bed  (per-donor); see
             _raw_data/bg4_consensus_by_manual_stage/PROVENANCE.md.
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

PATIENT_BED = Path('/scratche/slow/sblab/melidi01/Nader/Rebecca_pre_manuscript'
                   '/peak_analysis2/SEACR_consensus/BG4/patient_level')
DATA_DIR = SCRIPT_DIR / 'data'
PNG_DIR = SCRIPT_DIR / 'figures' / 'png'
SVG_DIR = SCRIPT_DIR / 'figures' / 'svg'
for d in (DATA_DIR, PNG_DIR, SVG_DIR):
    d.mkdir(parents=True, exist_ok=True)

config.setup_style()

# donor -> filename condition (for CONDITION_MAP lookup)
DONOR_COND = {
    'AHM1678': 'NG', 'CAM450': 'NG',
    'CAM408': 'ES', 'CAM412': 'ES', 'WTSI_OESO_146': 'ES', 'WTSI_OESO_117': 'LS',
    'CAM277': 'LS', 'CAM574': 'LS',
    'CAM428': 'ES_Rx', 'CAM629': 'LS_Rx',
}
STAGE_ORDER = ['NG', 'ES_EAC', 'LS_EAC', 'ES_Rx', 'LS_Rx']
PROGRESSION = ['NG', 'ES_EAC', 'LS_EAC']
LABELS = {'NG': 'NG', 'ES_EAC': 'ES', 'LS_EAC': 'LS', 'ES_Rx': 'ES+Rx', 'LS_Rx': 'LS+Rx'}


def count_bed(path):
    return sum(1 for _ in open(path)) if path.exists() else None


# ---- per-donor counts ------------------------------------------------------
rows = []
for donor, cond in DONOR_COND.items():
    stage = config.CONDITION_MAP.get((donor, cond))
    n = count_bed(PATIENT_BED / f'consensus_{donor}.bed')
    rows.append({'donor': donor, 'file_condition': cond, 'stage': stage,
                 'n_peaks': n})
per_donor = pd.DataFrame(rows).dropna(subset=['n_peaks'])
per_donor['n_peaks'] = per_donor['n_peaks'].astype(int)
per_donor.to_csv(DATA_DIR / 'fig1_peak_counts_per_donor.csv', index=False)

# reproducible (>=2-of-N donor) consensus set SIZE per stage — a distinct
# quantity from the per-donor median: the cross-donor reproducible peak set used
# as the G4 universe. (For n=1 stages it equals that single donor's count.)
BED_DIR = (SCRIPT_DIR / '..' / '_raw_data' / 'bg4_consensus_by_manual_stage').resolve()
CONSENSUS = {st: count_bed(BED_DIR / f'G4_repro2_{st}.bed') for st in STAGE_ORDER}

# stage summary (median + IQR) + the reproducible consensus size
summ = (per_donor.groupby('stage')['n_peaks']
        .agg(n_donors='count', median='median',
             q25=lambda x: np.percentile(x, 25),
             q75=lambda x: np.percentile(x, 75),
             min='min', max='max')
        .reindex(STAGE_ORDER).reset_index())
summ['reproducible_consensus_peaks'] = summ['stage'].map(CONSENSUS)
summ.to_csv(DATA_DIR / 'fig1_peak_counts_per_stage.csv', index=False)
print('per-donor counts:\n', per_donor.to_string(index=False))
print('\nstage summary (median + IQR):\n', summ.to_string(index=False))


# ---- plotting --------------------------------------------------------------
def dot_panel(stages, fname, title, panel_letter):
    fig, ax = plt.subplots(figsize=(config.COL_SINGLE * (1.1 if len(stages) == 3 else 1.7), 4.4))
    rng = np.random.RandomState(0)
    ymax = max(per_donor[per_donor['stage'].isin(stages)]['n_peaks'].max(),
               max(CONSENSUS[s] for s in stages))
    for xi, stage in enumerate(stages):
        col = config.CONDITION_COLORS.get(stage, '#888888')
        vals = per_donor[per_donor['stage'] == stage]['n_peaks'].values
        single = len(vals) < 2
        # median + IQR ONLY for multi-donor stages; single-donor (n=1) stages
        # show just their lone point — no median/diagonal line.
        if not single:
            q25, q75 = np.percentile(vals, [25, 75])
            ax.plot([xi, xi], [q25, q75], color=col, lw=2.2, zorder=2, solid_capstyle='round')
            ax.plot([xi - 0.16, xi + 0.16], [np.median(vals), np.median(vals)],
                    color='#222', lw=2.6, zorder=4)
        # per-donor points (jittered)
        jit = (rng.uniform(-0.10, 0.10, size=len(vals)) if len(vals) > 1 else np.zeros(len(vals)))
        ax.scatter(np.full(len(vals), xi) + jit, vals, s=130, color=col,
                   edgecolor='white' if not single else '#333',
                   linewidth=1.2 if not single else 1.6,
                   hatch=None, alpha=0.95, zorder=5)
        # reproducible (>=2-of-N donor) consensus SET SIZE — open diamond marker
        cval = CONSENSUS[stage]
        ax.scatter([xi], [cval], s=190, facecolors='none', edgecolors=col,
                   linewidths=2.2, marker='D', zorder=6)
        ax.annotate('{:,}'.format(cval), (xi, cval), xytext=(10, 0),
                    textcoords='offset points', va='center', ha='left',
                    fontsize=11, color=col, fontweight='bold')
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([LABELS[s] for s in stages])
    ax.set_ylabel('BG4 G4 peaks')
    # legend: per-donor dot vs reproducible consensus diamond
    from matplotlib.lines import Line2D
    leg_h = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#666',
               markersize=11, label='per-donor count'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='none',
               markeredgecolor='#666', markeredgewidth=2, markersize=11,
               label='≥2-of-N reproducible set'),
    ]
    ax.legend(handles=leg_h, loc='upper left', frameon=False, fontsize=10.5)
    ax.set_title(title)
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_ylim(0, ymax * 1.12)
    # n-donor annotation placed BELOW the x-tick labels (blended transform:
    # x in data coords, y in axes-fraction) so it never overlaps the bars/ticks.
    from matplotlib.transforms import blended_transform_factory
    tform = blended_transform_factory(ax.transData, ax.transAxes)
    for xi, stage in enumerate(stages):
        nstage = len(per_donor[per_donor['stage'] == stage])
        single = nstage < 2
        ax.text(xi, -0.11, 'n={}'.format(nstage) + ('*' if single else ''),
                transform=tform, ha='center', va='top', fontsize=12, color='#555')
    if any(len(per_donor[per_donor['stage'] == s]) < 2 for s in stages):
        ax.text(0.5, -0.235,
                '* single donor (n=1): point only (no median line, no IQR)',
                transform=ax.transAxes, ha='center', va='top', fontsize=11, color='#777')
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
    print(f'  saved {fname}')


dot_panel(PROGRESSION, 'fig1_A_peak_counts_progression',
          'BG4 G4 peaks per stage (progression)', 'A')
dot_panel(STAGE_ORDER, 'fig1_B_peak_counts_with_Rx',
          'BG4 G4 peaks per stage (incl. Rx)', 'B')

print('\nDone. Outputs in figures/png, figures/svg, data/.')
