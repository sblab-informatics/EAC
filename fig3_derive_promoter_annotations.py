#!/usr/bin/env python3
"""
Figure 3 — STEP 0: derive per-promoter G4 status + histone marks per stage.

Reproducible derivation (run this before the flow panels). Inputs are the
canonical peak sets copied into data/peak_sets/ (so Figure 3 is self-contained):

  BG4_G4_<stage>.bed      reproducible (>=2-of-N donor) BG4 G4 consensus,
                          manual classification. See
                          ../_raw_data/bg4_consensus_by_manual_stage/PROVENANCE.md
  H3K27ac_<stage>.bed     0.05fdr.stringent SEACR peaks, UNION of the stage's
  H3K4me1_<stage>.bed     donors (>=1 donor), manual classification, canonical
  H3K27me3_<stage>.bed    chroms. See
                          ../_raw_data/chromatin_consensus_by_manual_stage/PROVENANCE.md

  promoters = reference_data/annotation/gencode_v44_promoters.bed
              (gencode v44, +/-2 kb TSS, 38,841 promoters, GRCh38)

A promoter is G4+ / K27ac+ / K4me1+ / K27me3+ at a stage if it overlaps that
stage's peak set by >=1 bp (bedtools intersect -u).

Outputs:
  data/promoter_g4_status_corrected.csv     gene x g4_<stage>
  data/promoter_histone_marks_corrected.csv gene x {H3K27ac,H3K27me3,H3K4me1}_<stage>
"""
import sys
import subprocess
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
PEAKS = DATA_DIR / 'peak_sets'
PROM_BED = SCRIPT_DIR / 'reference_data' / 'annotation' / 'gencode_v44_promoters.bed'
STAGES = ['NG', 'ES_EAC', 'LS_EAC', 'ES_Rx', 'LS_Rx']


def genes_overlapping(peak_bed):
    """Gene symbols whose promoter overlaps peak_bed by >=1 bp."""
    res = subprocess.run(['bedtools', 'intersect', '-a', str(PROM_BED),
                          '-b', str(peak_bed), '-u'],
                         capture_output=True, text=True, check=True)
    return set(l.split('\t')[3] for l in res.stdout.splitlines() if l.strip())


prom = pd.read_csv(PROM_BED, sep='\t', header=None,
                   names=['chrom', 'start', 'end', 'gene', 'ensg', 'strand'])
print(f"promoters (gencode v44): {len(prom):,}")

# ---- G4 status -------------------------------------------------------------
g4 = prom.copy()
for st in STAGES:
    bed = PEAKS / f'BG4_G4_{st}.bed'
    pos = genes_overlapping(bed)
    g4[f'g4_{st}'] = g4['gene'].apply(lambda x: 'G4+' if x in pos else 'G4-')
    print(f"  G4  {st:7s}: G4+ promoters = {(g4[f'g4_{st}']=='G4+').sum():,}")
g4.to_csv(DATA_DIR / 'promoter_g4_status_corrected.csv', index=False)

# ---- histone marks ---------------------------------------------------------
marks = prom.copy()
for st in STAGES:
    for mark in ['H3K27ac', 'H3K27me3', 'H3K4me1']:
        pos = genes_overlapping(PEAKS / f'{mark}_{st}.bed')
        marks[f'{mark}_{st}'] = marks['gene'].isin(pos).astype(int)
    print(f"  mark {st:7s}: K27ac={marks[f'H3K27ac_{st}'].sum():,} "
          f"K27me3={marks[f'H3K27me3_{st}'].sum():,} K4me1={marks[f'H3K4me1_{st}'].sum():,}")
marks.to_csv(DATA_DIR / 'promoter_histone_marks_corrected.csv', index=False)

print('\nWrote promoter_g4_status_corrected.csv + promoter_histone_marks_corrected.csv')
