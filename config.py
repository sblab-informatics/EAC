#!/usr/bin/env python3
"""
config.py - Self-contained configuration for g4_chromatin_standalone

Five-condition analysis (no NDBE, no NA913):
  NG, ES_EAC, LS_EAC, ES_Rx, LS_Rx

Enhancer classification (from first principles):
  All enhancers MUST be ATAC+ (chromatin accessible)
  Active:  H3K27ac+ AND H3K4me1+ AND ATAC+
  Primed:  H3K4me1+ only (no H3K27ac, no H3K27me3) AND ATAC+
  Poised:  H3K4me1+ AND H3K27me3+ AND ATAC+

Nature-standard publication style:
  - No bar plots — Cleveland dots, lollipops, enrichment dots, rainclouds
  - Thin axes (0.5pt), despined, no figure titles
  - Panel labels A/B/C at standard position
  - 8pt ticks, 10pt labels, Arial font

All bigWig files are Drosophila spike-in normalized (Down_normalised/).
"""

import os
import re
import subprocess
import tempfile
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

warnings.filterwarnings('ignore')

# =============================================================================
# PARALLELIZATION
# =============================================================================

N_CPUS = os.cpu_count()

# =============================================================================
# PATHS — All relative to this project directory
# =============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUTPUT_DIR / 'data'
FIG_DIR = OUTPUT_DIR / 'figures'
FIGURE_DATA_DIR = DATA_DIR / 'figure_data'

# BigWig symlinks (point to original data)
BG4_BW_DIR = OUTPUT_DIR / 'bigwigs' / 'bg4'
ATAC_BW_DIR = OUTPUT_DIR / 'bigwigs' / 'atac'
HISTONE_BW_DIR = OUTPUT_DIR / 'bigwigs' / 'histone'

# Raw peak symlinks
BG4_RAW_PEAK_DIR = OUTPUT_DIR / 'raw_peaks' / 'bg4'
ATAC_RAW_PEAK_DIR = OUTPUT_DIR / 'raw_peaks' / 'atac'

# Reference data (copied locally)
REF_DATA_DIR = OUTPUT_DIR / 'reference_data'
PROMOTER_BED = REF_DATA_DIR / 'annotation' / 'gencode_v44_promoters.bed'
TSS_BED = REF_DATA_DIR / 'annotation' / 'gencode_v44_tss.bed'
HISTONE_CONSENSUS_DIR = REF_DATA_DIR / 'histone_marks'
GMT_DIR = REF_DATA_DIR / 'gene_sets'
DE_RESULTS_DIR = REF_DATA_DIR / 'de_results'

# RNA-seq data (copied locally)
TPM_MATRIX = OUTPUT_DIR / 'rnaseq_data' / 'gene_tpm_matrix.csv'
SAMPLE_METADATA = OUTPUT_DIR / 'rnaseq_data' / 'sample_metadata.csv'

# Master data (copied locally)
MASTER_GENE_ATLAS = DATA_DIR / 'master_gene_atlas.csv'
ENHANCER_ATLAS = DATA_DIR / '10_enhancer_atlas.csv'
ENHANCER_GENE_LINKAGE = DATA_DIR / '10_enhancer_gene_linkage.csv'
ENHANCER_LINKAGE = DATA_DIR / 'enhancer_gene_linkage.csv'

# BG4 consensus BEDs (rebuilt, unbinned — from raw SEACR peaks)
BG4_CONSENSUS_DIR = DATA_DIR / 'consensus_beds' / 'bg4'
BG4_PATIENT_DIR = DATA_DIR / 'consensus_beds' / 'bg4_patient'

# ATAC consensus BEDs (rebuilt from raw SEACR peaks)
ATAC_CONSENSUS_DIR = DATA_DIR / 'consensus_beds' / 'atac'

# Old v4 ATAC consensus (copied for reference)
V4_ATAC_CONSENSUS_DIR = REF_DATA_DIR / 'atac'

# Promoter G4 status (from unbinned consensus BEDs)
PROMOTER_G4_STATUS_PATH = DATA_DIR / 'promoter_g4_status.csv'

# =============================================================================
# CONDITIONS & SAMPLES
# =============================================================================

CONDITIONS = ['NG', 'ES_EAC', 'LS_EAC', 'ES_Rx', 'LS_Rx']
MAIN_CONDITIONS = ['NG', 'ES_EAC', 'LS_EAC']
SUPP_CONDITIONS = ['ES_Rx', 'LS_Rx']

CONDITION_COLORS = {
    'NG': '#2166AC',
    'ES_EAC': '#F4A582',
    'LS_EAC': '#B2182B',
    'ES_Rx': '#F4A582',
    'LS_Rx': '#B2182B',
}

CONDITION_HATCHES = {
    'ES_Rx': '///',
    'LS_Rx': '///',
}

CONDITION_EDGECOLORS = {
    'NG': '#2166AC',
    'ES_EAC': '#F4A582',
    'LS_EAC': '#B2182B',
    'ES_Rx': '#C06000',
    'LS_Rx': '#800000',
}

CONDITION_MARKERS = {
    'NG': 'o', 'ES_EAC': 'o', 'LS_EAC': 'o',
    'ES_Rx': 'D', 'LS_Rx': 'D',
}
CONDITION_FILLSTYLES = {
    'NG': 'full', 'ES_EAC': 'full', 'LS_EAC': 'full',
    'ES_Rx': 'none', 'LS_Rx': 'none',
}

CONDITION_LABELS = {
    'NG': 'NG', 'ES_EAC': 'ES', 'LS_EAC': 'LS',
    'ES_Rx': 'ES+Rx', 'LS_Rx': 'LS+Rx',
}

PATIENTS_BY_CONDITION = {
    'NG': ['AHM1678'],
    'ES_EAC': ['CAM408', 'CAM412', 'WTSI_OESO_146', 'WTSI_OESO_117'],
    'LS_EAC': ['CAM277', 'CAM574', 'CAM450'],
    'ES_Rx': ['CAM428'],
    'LS_Rx': ['CAM629'],
}

ALL_PATIENTS = [p for pts in PATIENTS_BY_CONDITION.values() for p in pts]

PATIENT_TO_CONDITION = {}
for cond, patients in PATIENTS_BY_CONDITION.items():
    for p in patients:
        PATIENT_TO_CONDITION[p] = cond

FILE_CONDITION_MAP = {
    'AHM1678': 'NG',
    'CAM408': 'ES',
    'WTSI_OESO_146': 'ES',
    'WTSI_OESO_117': 'LS',
    'CAM412': 'ES',
    'CAM277': 'LS',
    'CAM574': 'LS',
    'CAM450': 'NG',
    'CAM629': 'LS_Rx',
    'CAM428': 'ES_Rx',
}

CONDITION_MAP = {
    ('AHM1678', 'NG'): 'NG',
    ('CAM408', 'ES'): 'ES_EAC',
    ('CAM412', 'ES'): 'ES_EAC',
    ('WTSI_OESO_146', 'ES'): 'ES_EAC',
    ('WTSI_OESO_117', 'LS'): 'ES_EAC',
    ('CAM277', 'LS'): 'LS_EAC',
    ('CAM574', 'LS'): 'LS_EAC',
    ('CAM450', 'NG'): 'LS_EAC',
    ('CAM428', 'ES_Rx'): 'ES_Rx',
    ('CAM629', 'LS_Rx'): 'LS_Rx',
}

RNASEQ_CONDITION_MAP = {
    'NG': 'NG', 'ES': 'ES_EAC', 'LS': 'LS_EAC',
    'ES_Rx': 'ES_Rx', 'LS_Rx': 'LS_Rx',
}

EXCLUDED_SAMPLES = {'NA913'}
EXCLUDED_CONDITIONS = {'NDBE'}
HUMAN_CHROMS = {f'chr{i}' for i in range(1, 23)} | {'chrX', 'chrY'}

# =============================================================================
# COLORS & STYLE
# =============================================================================

METHOD_COLORS = {
    'GoPeaks': '#2166AC',
    'SEACR': '#B2182B',
    'Shared': '#7F7F7F',
}

DIRECTION_COLORS = {
    'Up': '#D73027',
    'Down': '#4575B4',
    'NS': '#CCCCCC',
}

ENHANCER_COLORS = {
    'Active': '#ff7f00',
    'Poised': '#4daf4a',
}

ENHANCER_TYPE_COLORS = {
    'Active': '#ff7f00',    # orange
    'Primed': '#984ea3',    # purple
    'Poised': '#4daf4a',    # green
}

G4_COLORS = {
    'G4+': '#d73027',
    'G4-': '#4575b4',
}

REPLICATE_COLORS = {
    'T1': '#2166AC',
    'T2': '#92C5DE',
    'T3': '#F4A582',
}

MARKS = ['H3K4me1', 'H3K27ac', 'H3K27me3', 'Pol2_S2_S5']

RX_HISTONE_MAP = {
    'ES_Rx': {
        'H3K4me1': 'NA931.organoid.CAM428.ES_Rx.H3K4me1.drosoSpiked_aH2Av.batch1.T1.merged.markduplicates.sortName.600.0.05fdr.stringent.bed',
        'H3K27ac': 'NA932.organoid.CAM428.ES_Rx.H3K27ac.drosoSpiked_aH2Av.batch1.T1.merged.markduplicates.sortName.600.0.05fdr.stringent.bed',
        'H3K27me3': 'NA933.organoid.CAM428.ES_Rx.H3K27me3.drosoSpiked_aH2Av.batch1.T1.merged.markduplicates.sortName.600.0.05fdr.stringent.bed',
        'Pol2_S2_S5': 'NA934.organoid.CAM428.ES_Rx.Pol2_S2_S5.drosoSpiked_aH2Av.batch1.T1.merged.markduplicates.sortName.600.0.05fdr.stringent.bed',
    },
    'LS_Rx': {
        'H3K4me1': 'NA955.organoid.CAM629.LS_Rx.H3K4me1.drosoSpiked_aH2Av.batch1.T1.merged.markduplicates.sortName.600.0.05fdr.stringent.bed',
        'H3K27ac': 'NA956.organoid.CAM629.LS_Rx.H3K27ac.drosoSpiked_aH2Av.batch1.T1.merged.markduplicates.sortName.600.0.05fdr.stringent.bed',
        'H3K27me3': 'NA957.organoid.CAM629.LS_Rx.H3K27me3.drosoSpiked_aH2Av.batch1.T1.merged.markduplicates.sortName.600.0.05fdr.stringent.bed',
        'Pol2_S2_S5': 'NA958.organoid.CAM629.LS_Rx.Pol2_S2_S5.drosoSpiked_aH2Av.batch1.T1.merged.markduplicates.sortName.600.0.05fdr.stringent.bed',
    },
}

ATAC_PATIENT_MAP = {
    'NG': ['NA986.organoid.AHM1678.NG.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed'],
    'ES_EAC': [
        'NA977.organoid.WTSI_OESO_146.ES.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed',
        'NA979.organoid.CAM408.ES.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed',
        'NA980.organoid.CAM412.ES.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed',
        'NA976.organoid.WTSI_OESO_117.LS.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed',
    ],
    'LS_EAC': [
        'NA978.organoid.CAM277.LS.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed',
        'NA983.organoid.CAM574.LS.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed',
        'NA982.organoid.CAM450.NG.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed',
    ],
    'ES_Rx': ['NA981.organoid.CAM428.ES_Rx.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed'],
    'LS_Rx': ['NA984.organoid.CAM629.LS_Rx.ATACseq..batch1.T1.merged.markduplicates.sortName.600..0.01fdr.stringent.bed'],
}

BIN_SIZE = 500

# =============================================================================
# NATURE-STANDARD STYLE
# =============================================================================

# Publication style: larger, cleaner type; Arial everywhere (10Jun build).
# Sizes bumped ~+60% over the prior Nature-small defaults for legibility at
# print size; Arial is locked as the sole family (no Helvetica/DejaVu fallback
# in the saved SVG/PDF text) per the manuscript style decision.
NATURE_RCPARAMS = {
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'font.size': 13,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'legend.title_fontsize': 13,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'figure.titlesize': 18,
    'figure.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'lines.linewidth': 1.4,
    'patch.linewidth': 0.6,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    # Render mathtext (log2, subscripts, chi^2, etc.) in the regular font
    # family (Arial) instead of matplotlib's DejaVu-based math font, so no
    # DejaVu leaks into the saved SVG/PDF text.
    'mathtext.default': 'regular',
    'mathtext.fontset': 'dejavusans',
}

COL_SINGLE = 3.35    # 85mm
COL_1_5 = 4.49       # 114mm
COL_FULL = 6.85      # 174mm


def _register_arial():
    """Ensure Arial is available to matplotlib in every (fresh) process.

    The 10Jun build ships an Arial family (Arimo, the Apache-2.0
    metric-compatible clone, renamed to 'Arial') under
    ~/.config/matplotlib/fonts/ttf so PNG previews and text-layout metrics
    use real Arial, and the saved SVG/PDF text carries font-family: Arial.
    This matplotlib build does not auto-scan that dir, so register it here;
    every figure script calls setup_style() before plotting. Idempotent.
    """
    import os
    import matplotlib.font_manager as fm
    fontdir = os.path.expanduser('~/.config/matplotlib/fonts/ttf')
    if not os.path.isdir(fontdir):
        return
    have = {f.name for f in fm.fontManager.ttflist}
    if 'Arial' in have:
        return
    for fp in fm.findSystemFonts(fontpaths=[fontdir]):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass


def setup_style():
    """Set publication style (10Jun: larger type, Arial everywhere)."""
    _register_arial()
    sns.set_theme(style='ticks', context='paper', font_scale=1.0)
    plt.rcParams.update(NATURE_RCPARAMS)
    # sns.set_theme resets rcParams, so re-assert the Arial family afterwards.
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.sans-serif'] = ['Arial']


# =============================================================================
# SVG FIX (Affinity Designer 2 compatibility)
# =============================================================================

def fix_svg_for_affinity(svg_text):
    """Fix matplotlib SVG for Affinity Designer 2 compatibility.

    Three fixes (canonical, vendored from ~/.local/bin/fix_svgs_for_affinity):
      1. CSS ``font:`` shorthand -> separate ``font-size:`` + ``font-family:``
         (matplotlib #22528; Affinity mis-parses the shorthand size/family).
      2. ``<use x= y=>`` -> ``<use transform="translate(x, y)">`` (#20910).
      3. strip ``<metadata>`` and ensure ``xmlns:xlink`` on the root <svg>.
    Idempotent. Assumes svg.fonttype='none' so text is editable <text>.
    """
    # --- 1. split CSS `font:` shorthand inside style="..." -------------------
    SIZE_KW = {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large',
               'xx-large', 'smaller', 'larger'}
    LENGTH_RE = re.compile(r'^\d[\d.]*(?:px|pt|em|rem|%|ex|ch|cm|mm|in|pc)$',
                           re.IGNORECASE)
    WEIGHT_KW = {'normal', 'bold', 'bolder', 'lighter',
                 '100', '200', '300', '400', '500', '600', '700', '800', '900'}
    STYLE_KW = {'italic', 'oblique'}
    VARIANT_KW = {'small-caps'}

    def _split_font_shorthand(m):
        parts = [p.strip() for p in m.group(1).split(';') if p.strip()]
        new = []
        for p in parts:
            if not p.startswith('font:'):
                new.append(p); continue
            val = p[5:].strip()
            tokens = val.split()
            size_idx = None
            for i, t in enumerate(tokens):
                t_main = t.split('/', 1)[0]
                if LENGTH_RE.match(t_main) or t_main.lower() in SIZE_KW:
                    size_idx = i; break
            if size_idx is None:
                sz = re.match(r'^(\S+)\s+(.+)$', val)
                if sz:
                    new.append('font-size: {}'.format(sz.group(1)))
                    new.append('font-family: {}'.format(sz.group(2)))
                else:
                    new.append(p)
                continue
            size_tok = tokens[size_idx]
            line_h = None
            if '/' in size_tok:
                size_tok, line_h = size_tok.split('/', 1)
            for tok in tokens[:size_idx]:
                low = tok.lower()
                if low in WEIGHT_KW:    new.append('font-weight: {}'.format(tok))
                elif low in STYLE_KW:   new.append('font-style: {}'.format(tok))
                elif low in VARIANT_KW: new.append('font-variant: {}'.format(tok))
            new.append('font-size: {}'.format(size_tok))
            if line_h:
                new.append('line-height: {}'.format(line_h))
            family = ' '.join(tokens[size_idx + 1:]).strip()
            if family:
                new.append('font-family: {}'.format(family))
        return 'style="' + '; '.join(new) + '"'
    svg_text = re.sub(r'style="([^"]*)"', _split_font_shorthand, svg_text)

    # --- 1b. collapse font-family fallback chain to Arial only ----------------
    # matplotlib emits `font-family: 'Arial', 'DejaVu Sans', 'Helvetica',
    # sans-serif`. Affinity renders the first (Arial) anyway, but per the
    # manuscript style we keep the saved text strictly Arial. If 'Arial' is in
    # the list, replace the whole family value with bare Arial.
    def _arial_only(m):
        fam = m.group(1).lower()
        # Collapse to Arial whenever the family is Arial, or matplotlib's
        # DejaVu/Helvetica/sans-serif fallbacks (incl. mathtext's bare
        # 'DejaVu Sans'). Leaves genuinely different families untouched.
        if ('arial' in fam or 'dejavu' in fam or 'helvetica' in fam
                or 'sans-serif' in fam):
            return 'font-family: Arial'
        return m.group(0)
    svg_text = re.sub(r"font-family:\s*([^;\"]*)", _arial_only, svg_text)

    # --- 2. <use x= y=> -> transform="translate(x, y)" -----------------------
    def replace_use(match):
        tag = match.group(0)
        x_m = re.search(r'\bx="([^"]*)"', tag)
        y_m = re.search(r'\by="([^"]*)"', tag)
        if not x_m and not y_m:
            return tag
        x = x_m.group(1) if x_m else "0"
        y = y_m.group(1) if y_m else "0"
        if x_m:
            tag = re.sub(r'\s*\bx="[^"]*"', '', tag)
        if y_m:
            tag = re.sub(r'\s*\by="[^"]*"', '', tag)
        tr = "translate({}, {})".format(x, y)
        existing = re.search(r'transform="([^"]*)"', tag)
        if existing:
            tag = re.sub(r'transform="[^"]*"',
                         'transform="{} {}"'.format(tr, existing.group(1)), tag)
        else:
            tag = re.sub(r'(\s*/>)', ' transform="{}"\\1'.format(tr), tag)
        return tag
    svg_text = re.sub(r'<use\b[^>]*(?:/>|>[^<]*</use>)', replace_use, svg_text)

    # --- 3. strip <metadata>, ensure xmlns:xlink -----------------------------
    svg_text = re.sub(r'\s*<metadata>.*?</metadata>\s*', '\n',
                      svg_text, flags=re.DOTALL)
    if 'xmlns:xlink' not in svg_text:
        svg_text = svg_text.replace(
            '<svg ',
            '<svg xmlns:xlink="http://www.w3.org/1999/xlink" ', 1)
    return svg_text

# =============================================================================
# FIGURE SAVE HELPERS
# =============================================================================

def save_figure(fig, name, script_num=None):
    """Save figure as PNG + SVG (with Affinity fix) + close."""
    png_dir = FIG_DIR / 'png'
    svg_dir = FIG_DIR / 'svg'
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)

    png_path = png_dir / '{}.png'.format(name)
    svg_path = svg_dir / '{}.svg'.format(name)

    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    with open(svg_path, 'r') as f:
        svg_text = f.read()
    fixed = fix_svg_for_affinity(svg_text)
    with open(svg_path, 'w') as f:
        f.write(fixed)

    plt.close(fig)
    print("  Saved: {} (.png + .svg)".format(name))


def save_pub(fig, name, script_num=None):
    """Save publication figure to publication/ directory."""
    pub_dir = FIG_DIR / 'publication'
    pub_dir.mkdir(parents=True, exist_ok=True)

    png_path = pub_dir / '{}.png'.format(name)
    svg_path = pub_dir / '{}.svg'.format(name)

    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(svg_path, format='svg', bbox_inches='tight',
                facecolor='white', edgecolor='none')

    with open(svg_path, 'r') as f:
        svg_text = f.read()
    fixed = fix_svg_for_affinity(svg_text)
    with open(svg_path, 'w') as f:
        f.write(fixed)

    plt.close(fig)
    print("  Saved (pub): {}".format(name))


def save_csv(df, name, subdir=None):
    """Save DataFrame as CSV."""
    out_dir = DATA_DIR / subdir if subdir else DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / '{}.csv'.format(name)
    df.to_csv(path, index=False)
    print("  CSV: {} ({} rows)".format(path.name, len(df)))


def save_figure_data(df, fig_name):
    """Save figure source data CSV."""
    FIGURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DATA_DIR / '{}_data.csv'.format(fig_name)
    df.to_csv(path, index=False)
    print("  Figure data: {}".format(path.name))

# =============================================================================
# ELEGANT PLOT HELPERS — No Bars
# =============================================================================

def panel_label(ax, letter, x=-0.05, y=1.05):
    """Add panel label (A, B, C, ...) at standard position."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=17, fontweight='bold', va='top', ha='right',
            fontfamily='Arial')


def cleveland_dot(ax, labels, values, colors=None, hatches=None,
                  marker_size=80, orientation='horizontal', edgecolors=None):
    """Cleveland dot plot — elegant replacement for bar charts."""
    n = len(labels)
    positions = np.arange(n)

    if colors is None:
        colors = [CONDITION_COLORS.get(l, '#333333') for l in labels]
    elif isinstance(colors, dict):
        colors = [colors.get(l, '#333333') for l in labels]

    if edgecolors is None:
        edgecolors = [CONDITION_EDGECOLORS.get(l, c) for l, c in zip(labels, colors)]
    elif isinstance(edgecolors, dict):
        edgecolors = [edgecolors.get(l, '#333333') for l in labels]

    if orientation == 'horizontal':
        for i, val in enumerate(values):
            ax.plot([0, val], [i, i], color='#cccccc', linewidth=0.5, zorder=1)

        for i, (label, val) in enumerate(zip(labels, values)):
            fc = colors[i]
            ec = edgecolors[i]
            marker = CONDITION_MARKERS.get(label, 'o')
            fill = CONDITION_FILLSTYLES.get(label, 'full')

            if fill == 'none':
                ax.scatter(val, i, s=marker_size, facecolors='white',
                          edgecolors=ec, linewidth=1.5, marker=marker, zorder=3)
            else:
                ax.scatter(val, i, s=marker_size, c=[fc],
                          edgecolors=ec, linewidth=0.5, marker=marker, zorder=3)

        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        for i, val in enumerate(values):
            ax.plot([i, i], [0, val], color='#cccccc', linewidth=0.5, zorder=1)

        for i, (label, val) in enumerate(zip(labels, values)):
            fc = colors[i]
            ec = edgecolors[i]
            marker = CONDITION_MARKERS.get(label, 'o')
            fill = CONDITION_FILLSTYLES.get(label, 'full')

            if fill == 'none':
                ax.scatter(i, val, s=marker_size, facecolors='white',
                          edgecolors=ec, linewidth=1.5, marker=marker, zorder=3)
            else:
                ax.scatter(i, val, s=marker_size, c=[fc],
                          edgecolors=ec, linewidth=0.5, marker=marker, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')

    sns.despine(ax=ax)


def lollipop(ax, labels, values, colors=None, orientation='horizontal',
             marker_size=60, stem_color='#999999', stem_width=0.8):
    """Lollipop chart — stem + dot, cleaner than bars."""
    n = len(labels)
    positions = np.arange(n)

    if colors is None:
        colors = [CONDITION_COLORS.get(l, '#333333') for l in labels]
    elif isinstance(colors, dict):
        colors = [colors.get(l, '#333333') for l in labels]

    if orientation == 'horizontal':
        ax.hlines(positions, 0, values, color=stem_color, linewidth=stem_width, zorder=1)
        ax.scatter(values, positions, c=colors, s=marker_size,
                  edgecolors='white', linewidth=0.5, zorder=3)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
    else:
        ax.vlines(positions, 0, values, color=stem_color, linewidth=stem_width, zorder=1)
        ax.scatter(positions, values, c=colors, s=marker_size,
                  edgecolors='white', linewidth=0.5, zorder=3)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')

    sns.despine(ax=ax)


def diverging_lollipop(ax, labels, gained, lost, gain_color='#D73027',
                       loss_color='#4575B4', marker_size=60):
    """Diverging lollipop — gained (right) and lost (left) from center."""
    n = len(labels)
    positions = np.arange(n)
    lost_neg = [-abs(v) for v in lost]

    ax.hlines(positions, 0, gained, color=gain_color, linewidth=0.8, zorder=1)
    ax.hlines(positions, 0, lost_neg, color=loss_color, linewidth=0.8, zorder=1)
    ax.scatter(gained, positions, c=gain_color, s=marker_size,
              edgecolors='white', linewidth=0.5, zorder=3)
    ax.scatter(lost_neg, positions, c=loss_color, s=marker_size,
              edgecolors='white', linewidth=0.5, zorder=3)

    ax.axvline(0, color='black', linewidth=0.5, zorder=2)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    sns.despine(ax=ax)


def _truncate_term(t, n=45):
    """Word-aware truncation for pathway/GO term names."""
    t = t.replace('HALLMARK_', '').replace('_', ' ')
    t = re.sub(r'\s*\(GO:\d+\)', '', t)
    if len(t) <= n:
        return t
    truncated = t[:n - 3]
    if ' ' in truncated:
        truncated = truncated.rsplit(' ', 1)[0]
    return truncated + '...'


def enrichment_dotplot(ax, terms, gene_ratio, pval, overlap_count,
                       cmap='YlOrRd', size_scale=5, max_terms=15,
                       max_chars=45, label_fontsize=12):
    """Nature-standard enrichment dot plot."""
    n = min(len(terms), max_terms)
    terms = terms[:n]
    gene_ratio = np.array(gene_ratio[:n])
    pval = np.array(pval[:n])
    overlap_count = np.array(overlap_count[:n])

    neg_log_p = -np.log10(np.clip(pval, 1e-50, None))
    positions = np.arange(n)

    display_terms = [_truncate_term(t, max_chars) for t in terms]

    scatter = ax.scatter(gene_ratio, positions, s=overlap_count * size_scale,
                        c=neg_log_p, cmap=cmap, edgecolors='#333333',
                        linewidth=0.3, zorder=3)

    ax.set_yticks(positions)
    ax.set_yticklabels(display_terms, fontsize=label_fontsize)
    ax.set_xlabel('Gene ratio')
    ax.invert_yaxis()

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.08)
    cbar.set_label('-log10(padj)', fontsize=12)
    cbar.ax.tick_params(labelsize=7)

    size_vals = np.array([5, 10, 20, 50])
    size_vals = size_vals[size_vals <= overlap_count.max() * 1.2]
    if len(size_vals) == 0:
        size_vals = np.array([overlap_count.min(), overlap_count.max()])
    legend_elements = [Line2D([0], [0], marker='o', color='w',
                              markerfacecolor='grey', markersize=np.sqrt(s * size_scale),
                              label=str(s)) for s in size_vals]
    leg = ax.legend(handles=legend_elements, title='Genes',
                   loc='lower right', fontsize=12, title_fontsize=12,
                   handletextpad=0.1, borderpad=0.3)

    sns.despine(ax=ax)
    return scatter


def raincloud(ax, data, group_col, value_col, order=None, palette=None,
              orient='v', width=0.6, point_size=2, alpha=0.6):
    """Raincloud plot — half-violin + jitter + box."""
    if order is None:
        order = [c for c in CONDITIONS if c in data[group_col].unique()]
    if palette is None:
        palette = CONDITION_COLORS

    for i, grp in enumerate(order):
        vals = data[data[group_col] == grp][value_col].dropna().values
        if len(vals) == 0:
            continue

        color = palette.get(grp, '#333333')

        if orient == 'v':
            parts = ax.violinplot([vals], positions=[i], showmeans=False,
                                  showmedians=False, showextrema=False,
                                  widths=width)
            for pc in parts['bodies']:
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                pc.get_paths()[0].vertices[:, 0] = np.clip(
                    pc.get_paths()[0].vertices[:, 0], m, None)
                pc.set_facecolor(color)
                pc.set_alpha(0.3)
                pc.set_edgecolor('none')

            bp = ax.boxplot([vals], positions=[i - width * 0.15], widths=width * 0.15,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color='black', linewidth=1),
                           boxprops=dict(facecolor=color, alpha=0.7, linewidth=0.5),
                           whiskerprops=dict(linewidth=0.5),
                           capprops=dict(linewidth=0.5))

            jitter = np.random.uniform(-width * 0.25, -width * 0.1, size=len(vals))
            ax.scatter(i + jitter, vals, s=point_size, c=color, alpha=alpha,
                      edgecolors='none', zorder=2)

            if grp in CONDITION_HATCHES:
                for patch in bp['boxes']:
                    patch.set_hatch(CONDITION_HATCHES[grp])
                    patch.set_edgecolor(CONDITION_EDGECOLORS.get(grp, 'black'))

        else:  # horizontal
            parts = ax.violinplot([vals], positions=[i], showmeans=False,
                                  showmedians=False, showextrema=False,
                                  widths=width, vert=False)
            for pc in parts['bodies']:
                m = np.mean(pc.get_paths()[0].vertices[:, 1])
                pc.get_paths()[0].vertices[:, 1] = np.clip(
                    pc.get_paths()[0].vertices[:, 1], m, None)
                pc.set_facecolor(color)
                pc.set_alpha(0.3)
                pc.set_edgecolor('none')

            bp = ax.boxplot([vals], positions=[i - width * 0.15], widths=width * 0.15,
                           patch_artist=True, showfliers=False, vert=False,
                           medianprops=dict(color='black', linewidth=1),
                           boxprops=dict(facecolor=color, alpha=0.7, linewidth=0.5),
                           whiskerprops=dict(linewidth=0.5),
                           capprops=dict(linewidth=0.5))

            jitter = np.random.uniform(-width * 0.25, -width * 0.1, size=len(vals))
            ax.scatter(vals, i + jitter, s=point_size, c=color, alpha=alpha,
                      edgecolors='none', zorder=2)

    if orient == 'v':
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in order])
    else:
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([CONDITION_LABELS.get(c, c) for c in order])
        ax.invert_yaxis()

    sns.despine(ax=ax)


def bubble_heatmap(ax, row_labels, col_labels, values, sizes,
                   cmap='RdBu_r', size_scale=300, vmin=None, vmax=None,
                   size_label='Count', color_label='Value'):
    """Bubble heatmap — sized dots in grid."""
    values = np.array(values)
    sizes = np.array(sizes)

    nr, nc = values.shape
    for i in range(nr):
        for j in range(nc):
            if sizes[i, j] > 0:
                ax.scatter(j, i,
                          s=sizes[i, j] / sizes.max() * size_scale,
                          c=[values[i, j]], cmap=cmap,
                          vmin=vmin or values.min(),
                          vmax=vmax or values.max(),
                          edgecolors='#333333', linewidth=0.3, zorder=3)

    ax.set_xticks(range(nc))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=12)
    ax.set_yticks(range(nr))
    ax.set_yticklabels(row_labels, fontsize=12)

    ax.set_xlim(-0.5, nc - 0.5)
    ax.set_ylim(nr - 0.5, -0.5)

    for i in range(nr + 1):
        ax.axhline(i - 0.5, color='#eeeeee', linewidth=0.3)
    for j in range(nc + 1):
        ax.axvline(j - 0.5, color='#eeeeee', linewidth=0.3)

    ax.set_aspect('equal')
    sns.despine(ax=ax, left=True, bottom=True)
    ax.tick_params(left=False, bottom=False)


def dumbbell(ax, labels, values1, values2, color1='#d73027', color2='#4575b4',
             label1='Group 1', label2='Group 2', marker_size=60):
    """Dumbbell plot — two dots connected by a line for each category."""
    n = len(labels)
    positions = np.arange(n)

    for i in range(n):
        ax.plot([values1[i], values2[i]], [i, i], color='#cccccc',
                linewidth=1, zorder=1)

    ax.scatter(values1, positions, c=color1, s=marker_size,
              edgecolors='white', linewidth=0.5, zorder=3, label=label1)
    ax.scatter(values2, positions, c=color2, s=marker_size,
              edgecolors='white', linewidth=0.5, zorder=3, label=label2)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.legend(fontsize=12, loc='lower right')
    sns.despine(ax=ax)


def stacked_area(ax, x_labels, data_dict, colors=None, alpha=0.7):
    """Stacked area chart — elegant replacement for stacked bars."""
    if colors is None:
        colors = {}

    categories = list(data_dict.keys())
    values = np.array([data_dict[c] for c in categories])
    x = np.arange(len(x_labels))

    ax.stackplot(x, values, labels=categories,
                colors=[colors.get(c, '#999999') for c in categories],
                alpha=alpha, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(fontsize=12, loc='upper left')
    ax.set_xlim(x[0], x[-1])
    sns.despine(ax=ax)


# =============================================================================
# STATISTICAL HELPERS
# =============================================================================

def p_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def add_stat_bracket(ax, x1, x2, y, pval, orient='v'):
    """Add statistical significance bracket between two positions."""
    stars = p_to_stars(pval)
    h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    if orient == 'v':
        ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h],
                lw=0.7, c='black')
        ax.text((x1 + x2) / 2, y, stars,
                ha='center', va='bottom', fontsize=12)


# =============================================================================
# DATA LOADING HELPERS
# =============================================================================

def load_tpm_matrix():
    """Load gene TPM matrix."""
    return pd.read_csv(TPM_MATRIX, index_col=0)


def load_sample_metadata():
    """Load sample metadata with reclassified conditions."""
    return pd.read_csv(SAMPLE_METADATA)


def load_de_results(comparison):
    """Load DESeq2 results for a given comparison."""
    path = DE_RESULTS_DIR / 'de_results_{}.csv'.format(comparison)
    if path.exists():
        return pd.read_csv(path)
    return None


def load_gmt(gmt_path):
    """Load GMT file as dict: {gene_set_name: set(genes)}."""
    gene_sets = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split('\t')
            name = parts[0]
            genes = set(parts[2:])
            gene_sets[name] = genes
    return gene_sets


def load_enhancer_linkage():
    """Load enhancer-gene linkage."""
    if ENHANCER_LINKAGE.exists():
        return pd.read_csv(ENHANCER_LINKAGE)
    return None


# =============================================================================
# BED / BEDTOOLS UTILITIES
# =============================================================================

def load_bed_peaks(filepath, filter_human=True):
    """Load peaks from a BED file as list of (chrom, start, end)."""
    peaks = []
    if not Path(filepath).exists():
        return peaks
    with open(filepath) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom = parts[0]
                if filter_human and chrom not in HUMAN_CHROMS:
                    continue
                peaks.append((chrom, int(parts[1]), int(parts[2])))
    return peaks


def peaks_to_bins(peaks, bin_size=BIN_SIZE):
    """Convert peaks to bins for set comparisons."""
    bins = set()
    for chrom, start, end in peaks:
        bin_start = (start // bin_size) * bin_size
        bin_end = ((end - 1) // bin_size) * bin_size
        for bin_pos in range(bin_start, bin_end + bin_size, bin_size):
            bins.add((chrom, bin_pos))
    return bins


def calculate_jaccard(bins1, bins2):
    """Jaccard index on bin sets."""
    if len(bins1) == 0 and len(bins2) == 0:
        return 0.0
    intersection = len(bins1 & bins2)
    union = len(bins1 | bins2)
    return intersection / union if union > 0 else 0.0


def filter_bed_human(bed_path, out_path):
    """Filter a BED file to human chromosomes only and sort."""
    chroms = '|'.join(sorted(HUMAN_CHROMS))
    cmd = "grep -E '^({})\t' {} | sort -k1,1 -k2,2n > {}".format(chroms, bed_path, out_path)
    subprocess.run(cmd, shell=True, check=True)


def merge_bed_files(bed_paths, out_path):
    """Cat multiple BED files, sort, merge into a single merged BED."""
    cat_parts = ' '.join(str(p) for p in bed_paths)
    cmd = "cat {} | sort -k1,1 -k2,2n | bedtools merge -i - > {}".format(cat_parts, out_path)
    subprocess.run(cmd, shell=True, check=True)


def count_peaks(bed_path):
    """Count lines (peaks) in a BED file."""
    result = subprocess.run(['wc', '-l', str(bed_path)], capture_output=True, text=True)
    return int(result.stdout.strip().split()[0])


def bedtools_intersect_count(bed_a, bed_b):
    """Count peaks in A that overlap with B."""
    cmd = "bedtools intersect -a {} -b {} -u | wc -l".format(bed_a, bed_b)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return int(result.stdout.strip())


def bedtools_intersect(bed_a, bed_b, flags='-u'):
    """Run bedtools intersect, return output lines."""
    cmd = "bedtools intersect -a {} -b {} {}".format(bed_a, bed_b, flags)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip().split('\n') if result.stdout.strip() else []


def bedtools_closest(bed_a, bed_b, flags='-d'):
    """Run bedtools closest, return output lines."""
    cmd = "bedtools closest -a {} -b {} {}".format(bed_a, bed_b, flags)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip().split('\n') if result.stdout.strip() else []


# =============================================================================
# BIGWIG SIGNAL HELPERS (must be module-level for ProcessPoolExecutor pickling)
# =============================================================================

def compute_auc_region(args):
    """Compute AUC for a single region from a bigWig. For parallel use."""
    bw_path, chrom, start, end = args
    try:
        import pyBigWig
        bw = pyBigWig.open(str(bw_path))
        vals = bw.values(chrom, start, end)
        bw.close()
        if vals is None:
            return 0.0
        arr = np.array(vals, dtype=float)
        arr = np.nan_to_num(arr, 0.0)
        return float(np.sum(arr))
    except Exception:
        return 0.0


def compute_mean_signal_region(args):
    """Compute mean signal for a single region from a bigWig. For parallel use."""
    bw_path, chrom, start, end = args
    try:
        import pyBigWig
        bw = pyBigWig.open(str(bw_path))
        vals = bw.values(chrom, start, end)
        bw.close()
        if vals is None:
            return 0.0
        arr = np.array(vals, dtype=float)
        arr = np.nan_to_num(arr, 0.0)
        return float(np.mean(arr))
    except Exception:
        return 0.0


def compute_signal_parallel(bw_path, regions, metric='auc'):
    """Compute signal across regions from a bigWig using all CPUs."""
    func = compute_auc_region if metric == 'auc' else compute_mean_signal_region
    args = [(bw_path, c, s, e) for c, s, e in regions]

    with ProcessPoolExecutor(max_workers=N_CPUS) as executor:
        results = list(executor.map(func, args, chunksize=500))

    return np.array(results)


# =============================================================================
# SAMPLE REGISTRY BUILDER
# =============================================================================

def build_sample_registry():
    """Build a comprehensive sample registry mapping all bigWig paths.

    Returns DataFrame with columns:
        na_id, patient, file_condition, condition, assay, rep, bw_path
    """
    records = []

    # BG4 bigWigs (spike-in normalized)
    for bw in sorted(BG4_BW_DIR.glob('*.down.merged..600.cpm.bs5.bw')):
        parts = bw.name.split('.')
        na_id = parts[0]
        if na_id in EXCLUDED_SAMPLES:
            continue
        patient = parts[2]
        file_cond = parts[3]
        if file_cond in EXCLUDED_CONDITIONS:
            continue
        key = (patient, file_cond)
        if key not in CONDITION_MAP:
            continue
        rep = next((p for p in parts if p in ('T1', 'T2', 'T3')), 'T1')
        records.append({
            'na_id': na_id, 'patient': patient, 'file_condition': file_cond,
            'condition': CONDITION_MAP[key], 'assay': 'BG4', 'rep': rep,
            'bw_path': str(bw),
        })

    # IgG bigWigs
    for bw in sorted(HISTONE_BW_DIR.glob('NA*.IgG*.merged..600.cpm.bs5.bw')):
        if 'markduplicates' in bw.name:
            continue
        parts = bw.name.split('.')
        na_id = parts[0]
        patient = parts[2]
        file_cond = parts[3]
        if file_cond in EXCLUDED_CONDITIONS:
            continue
        key = (patient, file_cond)
        if key not in CONDITION_MAP:
            continue
        rep = next((p for p in parts if p in ('T1', 'T2', 'T3')), 'T1')
        records.append({
            'na_id': na_id, 'patient': patient, 'file_condition': file_cond,
            'condition': CONDITION_MAP[key], 'assay': 'IgG', 'rep': rep,
            'bw_path': str(bw),
        })

    # ATAC bigWigs
    for bw in sorted(ATAC_BW_DIR.glob('NA*.ATACseq*.merged..600.cpm.bs5.bw')):
        if 'markduplicates' in bw.name:
            continue
        parts = bw.name.split('.')
        na_id = parts[0]
        patient = parts[2]
        file_cond = parts[3]
        if file_cond in EXCLUDED_CONDITIONS:
            continue
        key = (patient, file_cond)
        if key not in CONDITION_MAP:
            continue
        rep = next((p for p in parts if p in ('T1', 'T2', 'T3')), 'T1')
        records.append({
            'na_id': na_id, 'patient': patient, 'file_condition': file_cond,
            'condition': CONDITION_MAP[key], 'assay': 'ATAC', 'rep': rep,
            'bw_path': str(bw),
        })

    # Histone mark bigWigs
    for mark in MARKS:
        for bw in sorted(HISTONE_BW_DIR.glob('NA*.{}*.merged..600.cpm.bs5.bw'.format(mark))):
            if 'markduplicates' in bw.name:
                continue
            parts = bw.name.split('.')
            na_id = parts[0]
            patient = parts[2]
            file_cond = parts[3]
            if file_cond in EXCLUDED_CONDITIONS:
                continue
            key = (patient, file_cond)
            if key not in CONDITION_MAP:
                continue
            rep = next((p for p in parts if p in ('T1', 'T2', 'T3')), 'T1')
            records.append({
                'na_id': na_id, 'patient': patient, 'file_condition': file_cond,
                'condition': CONDITION_MAP[key], 'assay': mark, 'rep': rep,
                'bw_path': str(bw),
            })

    df = pd.DataFrame(records)
    return df


def load_registry(registry_path=None):
    """Load or build sample registry."""
    if registry_path is None:
        registry_path = DATA_DIR / 'sample_registry.csv'
    if registry_path.exists():
        return pd.read_csv(registry_path)
    reg = build_sample_registry()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    reg.to_csv(registry_path, index=False)
    print("  Built sample registry: {} entries".format(len(reg)))
    return reg


# =============================================================================
# ENRICHMENT ANALYSIS
# =============================================================================

def fisher_enrichment(gene_set, target_genes, background_genes):
    """Fisher's exact test for gene set enrichment."""
    from scipy.stats import fisher_exact

    target = set(target_genes) & set(background_genes)
    bg = set(background_genes) - target
    gs = set(gene_set) & set(background_genes)

    a = len(target & gs)
    b = len(target - gs)
    c = len(bg & gs)
    d = len(bg - gs)

    odds, pval = fisher_exact([[a, b], [c, d]], alternative='greater')
    return odds, pval


def run_enrichment(gene_list, gmt_path, background=None, max_terms=20):
    """Run gene set enrichment analysis using Fisher's exact test."""
    from scipy.stats import false_discovery_control

    gene_sets = load_gmt(gmt_path)
    query = set(gene_list)

    if background is not None:
        bg = set(background)
    else:
        bg = set()
        for gs in gene_sets.values():
            bg |= gs
        bg |= query

    query_in_bg = len(query & bg)

    results = []
    for term, gs in gene_sets.items():
        gs_in_bg = gs & bg
        if len(gs_in_bg) < 5:
            continue
        overlap_genes = query & gs_in_bg
        if len(overlap_genes) == 0:
            continue
        odds, pval = fisher_enrichment(gs_in_bg, query, bg)
        gene_ratio = len(overlap_genes) / query_in_bg if query_in_bg > 0 else 0
        results.append({
            'term': term,
            'overlap': len(overlap_genes),
            'gene_set_size': len(gs_in_bg),
            'query_size': query_in_bg,
            'gene_ratio': gene_ratio,
            'pvalue': pval,
            'odds_ratio': odds,
            'genes': ';'.join(sorted(overlap_genes)),
        })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values('pvalue')

    if len(df) > 1:
        df['padj'] = false_discovery_control(df['pvalue'].values, method='bh')
    else:
        df['padj'] = df['pvalue']

    return df.head(max_terms)


def run_enrichment_gseapy(gene_list, gmt_path, background=None, max_terms=20, cutoff=1.0):
    """Run ORA via gseapy.enrich() — proper Fisher's test with FDR."""
    import gseapy

    if len(gene_list) < 5 or not Path(gmt_path).exists():
        return pd.DataFrame()

    try:
        kw = dict(gene_list=list(gene_list),
                  gene_sets=str(gmt_path),
                  outdir=None, no_plot=True, cutoff=cutoff)
        if background is not None and len(background) > 0:
            kw['background'] = list(background)
        enr = gseapy.enrich(**kw)
        df = enr.results.copy()

        df = df.rename(columns={
            'Term': 'term',
            'Adjusted P-value': 'padj',
            'P-value': 'pvalue',
            'Odds Ratio': 'odds_ratio',
            'Genes': 'genes',
            'Overlap': 'overlap_str',
            'Combined Score': 'combined_score',
        })

        if 'overlap_str' in df.columns:
            df['overlap'] = df['overlap_str'].apply(
                lambda x: int(str(x).split('/')[0]))
            df['gene_set_size'] = df['overlap_str'].apply(
                lambda x: int(str(x).split('/')[1]))

        df['query_size'] = len(gene_list)
        df['gene_ratio'] = df['overlap'] / df['query_size']
        df = df.sort_values('padj')
        return df.head(max_terms)

    except Exception as e:
        print("  ORA error: {}".format(e))
        return pd.DataFrame()


# =============================================================================
# CONDITION LEGEND HELPER
# =============================================================================

def make_condition_legend(ax, conditions=None, loc='upper right'):
    """Create a legend showing condition colors with correct markers."""
    if conditions is None:
        conditions = CONDITIONS

    handles = []
    for cond in conditions:
        color = CONDITION_COLORS[cond]
        marker = CONDITION_MARKERS.get(cond, 'o')
        fill = CONDITION_FILLSTYLES.get(cond, 'full')
        label = CONDITION_LABELS.get(cond, cond)

        if fill == 'none':
            h = Line2D([0], [0], marker=marker, color='w',
                      markerfacecolor='white',
                      markeredgecolor=CONDITION_EDGECOLORS.get(cond, color),
                      markeredgewidth=1.5, markersize=7, label=label)
        else:
            h = Line2D([0], [0], marker=marker, color='w',
                      markerfacecolor=color,
                      markeredgecolor=CONDITION_EDGECOLORS.get(cond, color),
                      markersize=7, label=label)
        handles.append(h)

    ax.legend(handles=handles, loc=loc, fontsize=12, frameon=False)


# =============================================================================
# ENHANCER RECLASSIFICATION
# =============================================================================

def _bed_to_set(bed_path, filter_human=True):
    """Load BED as set of (chrom, start, end) tuples."""
    regions = set()
    if not Path(bed_path).exists():
        print("  WARNING: BED not found: {}".format(bed_path))
        return regions
    with open(bed_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            chrom = parts[0]
            if filter_human and chrom not in HUMAN_CHROMS:
                continue
            regions.add((chrom, int(parts[1]), int(parts[2])))
    return regions


def _intersect_beds(bed_a_regions, bed_b_path, min_overlap_frac=1e-9):
    """Check which regions in bed_a overlap with bed_b using bedtools."""
    if not Path(bed_b_path).exists():
        return set()

    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as tmp:
        for chrom, start, end in sorted(bed_a_regions):
            tmp.write('{}\t{}\t{}\n'.format(chrom, start, end))
        tmp_path = tmp.name

    try:
        cmd = "sort -k1,1 -k2,2n {} | bedtools intersect -a - -b {} -u".format(
            tmp_path, bed_b_path)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        overlapping = set()
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                parts = line.split('\t')
                overlapping.add((parts[0], int(parts[1]), int(parts[2])))
        return overlapping
    finally:
        os.unlink(tmp_path)


def _subtract_beds(bed_a_regions, bed_b_path):
    """Return regions in bed_a that do NOT overlap bed_b."""
    if not Path(bed_b_path).exists():
        return bed_a_regions

    overlapping = _intersect_beds(bed_a_regions, bed_b_path)
    return bed_a_regions - overlapping


def classify_enhancers_for_condition(condition):
    """Classify BG4 peaks for a condition into enhancer types.

    Steps:
    1. Load BG4 consensus peaks for the condition
    2. Remove promoter-overlapping peaks
    3. Keep only ATAC+ peaks (chromatin accessible)
    4. Classify by histone marks:
       Active:  H3K27ac+ AND H3K4me1+
       Primed:  H3K4me1+ only (no H3K27ac, no H3K27me3)
       Poised:  H3K4me1+ AND H3K27me3+
    """
    bg4_bed = BG4_CONSENSUS_DIR / '{}_consensus.bed'.format(condition)
    atac_bed = ATAC_CONSENSUS_DIR / 'ATAC_{}_consensus.bed'.format(condition)
    h3k27ac_bed = HISTONE_CONSENSUS_DIR / 'H3K27ac_{}_merged.bed'.format(condition)
    h3k4me1_bed = HISTONE_CONSENSUS_DIR / 'H3K4me1_{}_merged.bed'.format(condition)
    h3k27me3_bed = HISTONE_CONSENSUS_DIR / 'H3K27me3_{}_merged.bed'.format(condition)

    print("  Classifying enhancers for {} ...".format(condition))

    bg4_peaks = _bed_to_set(bg4_bed)
    print("    BG4 peaks: {}".format(len(bg4_peaks)))

    non_promoter_bg4 = _subtract_beds(bg4_peaks, str(PROMOTER_BED))
    print("    Non-promoter BG4: {}".format(len(non_promoter_bg4)))

    atac_pos_bg4 = _intersect_beds(non_promoter_bg4, str(atac_bed))
    print("    ATAC+ non-promoter BG4: {}".format(len(atac_pos_bg4)))

    has_h3k27ac = _intersect_beds(atac_pos_bg4, str(h3k27ac_bed))
    has_h3k4me1 = _intersect_beds(atac_pos_bg4, str(h3k4me1_bed))
    has_h3k27me3 = _intersect_beds(atac_pos_bg4, str(h3k27me3_bed))

    active = has_h3k27ac & has_h3k4me1
    poised = (has_h3k4me1 & has_h3k27me3) - active
    primed = has_h3k4me1 - active - poised
    unclassified = atac_pos_bg4 - active - primed - poised

    print("    Active (G4+): {}".format(len(active)))
    print("    Primed (G4+): {}".format(len(primed)))
    print("    Poised (G4+): {}".format(len(poised)))
    print("    Unclassified: {}".format(len(unclassified)))

    # G4- enhancers
    atac_peaks = _bed_to_set(atac_bed)
    non_promoter_atac = _subtract_beds(atac_peaks, str(PROMOTER_BED))
    g4_neg_atac = non_promoter_atac - _intersect_beds(non_promoter_atac, str(bg4_bed))
    print("    ATAC+ non-promoter non-BG4 (G4-): {}".format(len(g4_neg_atac)))

    g4neg_h3k27ac = _intersect_beds(g4_neg_atac, str(h3k27ac_bed))
    g4neg_h3k4me1 = _intersect_beds(g4_neg_atac, str(h3k4me1_bed))
    g4neg_h3k27me3 = _intersect_beds(g4_neg_atac, str(h3k27me3_bed))

    g4neg_active = g4neg_h3k27ac & g4neg_h3k4me1
    g4neg_poised = (g4neg_h3k4me1 & g4neg_h3k27me3) - g4neg_active
    g4neg_primed = g4neg_h3k4me1 - g4neg_active - g4neg_poised

    print("    Active (G4-): {}".format(len(g4neg_active)))
    print("    Primed (G4-): {}".format(len(g4neg_primed)))
    print("    Poised (G4-): {}".format(len(g4neg_poised)))

    return {
        'Active': sorted(active),
        'Primed': sorted(primed),
        'Poised': sorted(poised),
        'Unclassified': sorted(unclassified),
        'g4_neg_Active': sorted(g4neg_active),
        'g4_neg_Primed': sorted(g4neg_primed),
        'g4_neg_Poised': sorted(g4neg_poised),
    }


def save_enhancer_beds(classified, condition, out_dir=None):
    """Save classified enhancer regions as BED files."""
    if out_dir is None:
        out_dir = DATA_DIR / 'enhancer_beds'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for etype, regions in classified.items():
        if not regions:
            continue
        g4_tag = 'G4neg' if etype.startswith('g4_neg_') else 'G4pos'
        clean_type = etype.replace('g4_neg_', '')
        fname = '{}_{}_{}_{}.bed'.format(clean_type, condition, g4_tag,
                                          len(regions))
        path = out_dir / fname
        with open(path, 'w') as f:
            for chrom, start, end in regions:
                f.write('{}\t{}\t{}\n'.format(chrom, start, end))
        print("  BED: {} ({} regions)".format(fname, len(regions)))


def load_enhancer_beds(condition, enhancer_type='Active', g4_status='G4pos'):
    """Load previously saved enhancer BED for a condition/type/G4 status."""
    bed_dir = DATA_DIR / 'enhancer_beds'
    pattern = '{}_{}_{}*.bed'.format(enhancer_type, condition, g4_status)
    matches = list(bed_dir.glob(pattern))
    if not matches:
        print("  WARNING: No BED found for pattern {}".format(pattern))
        return []
    bed_path = matches[0]
    return load_bed_peaks(str(bed_path))


# =============================================================================
# CONVENIENCE LOADERS
# =============================================================================

def load_master_gene_atlas():
    """Load master gene atlas, replacing G4 status with unbinned consensus classification."""
    atlas = pd.read_csv(MASTER_GENE_ATLAS)

    if PROMOTER_G4_STATUS_PATH.exists():
        g4_df = pd.read_csv(PROMOTER_G4_STATUS_PATH)
        g4_cols = [c for c in g4_df.columns if c.startswith('g4_')]
        gene_g4 = g4_df.groupby('gene')[g4_cols].apply(
            lambda x: x.apply(lambda col: 'G4+' if (col == 'G4+').any() else 'G4-')
        ).reset_index()
        old_g4_cols = [c for c in atlas.columns if c.startswith('g4_')]
        atlas = atlas.drop(columns=old_g4_cols, errors='ignore')
        atlas = atlas.merge(gene_g4, on='gene', how='left')
        for col in g4_cols:
            if col in atlas.columns:
                atlas[col] = atlas[col].fillna('G4-')

    return atlas


def load_enhancer_atlas_v4():
    """Load the enhancer atlas."""
    return pd.read_csv(ENHANCER_ATLAS)


def load_enhancer_linkage_v4():
    """Load the enhancer-gene linkage."""
    return pd.read_csv(ENHANCER_GENE_LINKAGE)


def load_promoter_bed():
    """Load promoter BED as DataFrame."""
    return pd.read_csv(PROMOTER_BED, sep='\t', header=None,
                       names=['chrom', 'start', 'end', 'gene', 'ensg', 'strand'])


_promoter_g4_cache = None

def load_promoter_g4_status():
    """Load promoter G4 status derived from unbinned BG4 consensus BEDs."""
    global _promoter_g4_cache
    if _promoter_g4_cache is None:
        _promoter_g4_cache = pd.read_csv(PROMOTER_G4_STATUS_PATH)
    return _promoter_g4_cache.copy()


def get_promoter_coords(genes, promoter_df=None):
    """Get promoter coordinates for a list of genes."""
    if promoter_df is None:
        promoter_df = load_promoter_bed()
    matched = promoter_df[promoter_df['gene'].isin(set(genes))]
    return list(zip(matched['chrom'], matched['start'], matched['end']))


def compute_condition_signal(regions, assay, condition, registry=None, metric='mean'):
    """Compute mean signal across all patients in a condition for an assay."""
    if registry is None:
        registry = load_registry()

    mask = (registry['assay'] == assay) & (registry['condition'] == condition)
    bw_paths = registry[mask]['bw_path'].tolist()

    if not bw_paths:
        print("  WARNING: No bigWigs for {} / {}".format(assay, condition))
        return np.zeros(len(regions))

    signals = []
    for bw_path in bw_paths:
        sig = compute_signal_parallel(bw_path, regions, metric=metric)
        signals.append(sig)

    return np.mean(signals, axis=0)


# Initialize style on import
setup_style()
print("g4_chromatin_standalone config loaded.")
print("  Output: {}".format(OUTPUT_DIR))


# --- effect-size-first reporting helpers (10Jun) -----------------------------
# Rationale: across n = thousands of genes/peaks the p-value mostly reads out
# sample size, not biological certainty, and underflows below the float64 floor.
# Lead with the effect size + 95% CI; report p only as a bounded floor.
_P_FLOOR = 1e-300

def spearman_ci(rho, n, alpha=0.05):
    """95% CI for Spearman rho via the Fisher-z transform. Returns (lo, hi)."""
    import numpy as _np
    from scipy import stats as _stats
    if rho is None or n is None or not _np.isfinite(rho) or n < 4:
        return (float('nan'), float('nan'))
    r = max(-0.999999999, min(0.999999999, float(rho)))
    z = _np.arctanh(r)
    se = 1.0 / _np.sqrt(n - 3)
    zc = _stats.norm.ppf(1 - alpha / 2)
    return (float(_np.tanh(z - zc * se)), float(_np.tanh(z + zc * se)))

def fmt_p_bounded(p):
    """Format a p-value for display: never an unbelievable exact tiny value.

    p below the float64 floor (or literal 0) -> 'p < 1e-300'. Otherwise the
    usual 'p = 1.2e-08'. Use on every figure label so the exact 10^-1985
    magnitude (which only reflects huge n) is never printed.
    """
    import numpy as _np
    try:
        p = float(p)
    except (TypeError, ValueError):
        return 'p = n/a'
    if not _np.isfinite(p) or p <= _P_FLOOR:
        return 'p < 1e-300'
    return 'p = {:.1e}'.format(p)


# --- effect-size heatmap with significance stars (pre_pub convention) ---
def stars_for_q(q):
    """FDR star tier: * <0.05, ** <0.01, *** <0.001, **** <1e-10."""
    if q is None or not (q == q):
        return ''
    if q < 1e-10: return '****'
    if q < 1e-3:  return '***'
    if q < 1e-2:  return '**'
    if q < 5e-2:  return '*'
    return ''


def effect_heatmap(ax, matrix, row_labels, col_labels, star_q=None,
                   cmap='RdBu_r', center=0.0, vmax=None, cbar_label='log2 effect',
                   scale='symmetric'):
    """COLOUR = effect size; optional STARS from `star_q` (same shape).

    scale='symmetric' (default): diverging, centred at `center`, range ±max|effect|
        (vmax overridable). Back-compatible — existing panels unchanged.
    scale='data': colour spans the heatmap's TRUE data range [min, max] (not
        symmetric); returns (im, vmin, vmax) so the caller can label the colorbar
        ends with the actual values. Use TwoSlopeNorm only if the range straddles
        `center`; otherwise a plain linear Normalize across [min, max].
    """
    import numpy as _np
    from matplotlib.colors import TwoSlopeNorm, Normalize
    M = _np.asarray(matrix, float)
    if scale == 'data':
        vmin = float(_np.nanmin(M)); vmax_ = float(_np.nanmax(M))
        if vmin == vmax_:
            vmin -= 0.5; vmax_ += 0.5
        if vmin < center < vmax_:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax_)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax_)
        ref = max(abs(vmin - center), abs(vmax_ - center)) or 1.0
    else:
        if vmax is None:
            vmax = _np.nanmax(_np.abs(M - center)) or 1.0
        norm = TwoSlopeNorm(vmin=center - vmax, vcenter=center, vmax=center + vmax)
        vmin, vmax_, ref = center - vmax, center + vmax, vmax
    im = ax.imshow(M, aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=13)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=10)
    if star_q is not None:
        Q = _np.asarray(star_q, float)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                s = stars_for_q(Q[i, j])
                if s:
                    col = 'white' if abs(M[i, j] - center) > 0.55 * ref else '#222'
                    ax.text(j, i, s, ha='center', va='center', fontsize=11,
                            fontweight='bold', color=col)
    if scale == 'data':
        return im, vmin, vmax_
    return im


def fix_svg_file(path):
    """Read an SVG file, apply fix_svg_for_affinity, write it back. Safe
    (reads fully before truncating). Use after fig.savefig(path, format='svg')."""
    with open(path) as _fh:
        _txt = _fh.read()
    with open(path, 'w') as _fh:
        _fh.write(fix_svg_for_affinity(_txt))
