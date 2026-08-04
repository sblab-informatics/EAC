# Figure 3 — Methods

## Promoter G-quadruplex and chromatin-state dynamics across EAC progression

Figure 3 maps the acquisition of promoter G-quadruplexes (G4s) and the
co-evolving chromatin state of promoters across oesophageal adenocarcinoma (EAC)
progression (normal squamous/glandular epithelium, NG → early EAC, ES_EAC → late
EAC, LS_EAC), and relates G4-gain to gene function, cancer-hallmark programmes
and named EAC driver genes. All analyses were performed in Python 3.8.8 using
bedtools v2.26.0 [ref], NumPy, pandas, SciPy (`scipy.stats`) and Matplotlib;
gene-set over-representation used the Fisher exact test as implemented in
`scipy.stats.fisher_exact`.

### Sample classification and consensus peak sets (shared upstream)

Cell lines were assigned to progression stages by manual classification from a
fixed configuration map (`CONDITION_MAP`), not from filename labels; the two
re-assignments were CAM450 → LS_EAC and WTSI_OESO_117 → ES_EAC. Donors per stage
were NG = 1 (AHM1678), ES_EAC = 4 (CAM408, CAM412, WTSI_OESO_146, WTSI_OESO_117),
LS_EAC = 3 (CAM277, CAM574, CAM450), with the two treated stages ES_Rx (CAM428)
and LS_Rx (CAM629) at n = 1 each. NA913 was excluded on QC grounds and the NDBE
(Barrett's) condition was excluded from the progression analysis. Analyses in
Figure 3 used the three progression stages NG, ES_EAC and LS_EAC.

Three canonical peak sets, copied locally so the figure is self-contained
(`data/peak_sets/`), underlie every panel:

- **BG4 G4 consensus** (`BG4_G4_<stage>.bed`): reproducible per-stage G4 calls
  built by requiring a peak in ≥2 of 3 BG4 technical replicates per donor and
  then a call in ≥2-of-N donors per stage; single-donor stages (NG, ES_Rx,
  LS_Rx) are that donor (n = 1). Peak counts were NG = 9,393, ES_EAC = 15,318,
  LS_EAC = 17,078.
- **Histone marks** (`H3K27ac_<stage>.bed`, `H3K4me1_<stage>.bed`,
  `H3K27me3_<stage>.bed`): SEACR `0.05fdr.stringent` peaks taken as the union of
  the stage's donors (marked in ≥1 donor). The union (rather than a ≥2-of-N
  reproducibility filter) was used because each donor contributed a single
  histone technical replicate; this build reproduces the original
  K27ac+ promoter counts (e.g. LS_EAC = 17,651 vs 17,587 previously).
- **Promoter annotation** (`gencode_v44_promoters.bed`): GENCODE v44 promoters
  defined as ±2 kb around each TSS, 38,841 promoters, GRCh38, restricted to the
  canonical human chromosomes (chr1–22, X, Y).

RNA-seq TPM (`data/rnaseq/gene_tpm_matrix.csv`) with the corrected sample
metadata (`sample_metadata_corrected.csv`, `analytical_condition` field) provided
per-stage expression; technical replicates were averaged to a stage-mean TPM per
gene.

### Per-promoter G4 and histone-mark annotation (`fig3_derive_promoter_annotations.py`)

For each stage, a promoter was scored G4+, H3K27ac+, H3K4me1+ or H3K27me3+ if it
overlapped the corresponding stage peak set by ≥1 bp (`bedtools intersect -u`,
GENCODE v44 promoters as A). This produced two per-promoter tables keyed on gene
symbol: a G4-status matrix (`promoter_g4_status_corrected.csv`; `g4_<stage>` ∈
{G4+, G4−}) and a histone-mark matrix (`promoter_histone_marks_corrected.csv`;
binary `{H3K27ac,H3K27me3,H3K4me1}_<stage>`). The G4+ promoter fraction rose
across progression, from 24.6 % (NG) to 35.2 % (ES_EAC) to 37.6 % (LS_EAC).

### Promoter G4-status flow (`fig3_status_flow.py`, Panel L)

Per-promoter G4 status was visualised as a thick-stack alluvial across NG → ES →
LS (G4+ red `#d73027`, G4− blue `#4575b4`), with within-stack counts and
percentages and per-transition annotation of promoters gaining (+) and losing
(−) a G4. Three promoter sets were plotted, each with its own source-data table
(`*_rawdata.csv`): all 38,841 GENCODE v44 promoters; promoters of genes with
stage-mean TPM > 0.1 in all three stages (n = 12,178); and promoters of genes
with TPM > 0.1 in any stage (n = 14,593). Transition tables tabulating gained,
lost, stable-G4+, stable-G4− and net-gain counts for NG→ES, ES→LS and NG→LS were
written for each set. Flow was dominated by G4 gain (all promoters, NG→LS:
+5,626 gained / −583 lost). This panel is descriptive; no inferential test was
applied to the flow itself.

### G4+ enrichment per chromatin state (`fig3_g4_increase.py`, Panel A2)

Promoters were assigned a chromatin state at each stage from their histone marks
under four classification schemes (see below). For each state and stage we
computed the number and percentage of promoters in that state that were G4+
(`pct_g4p` = 100 × n_G4+ / n_state) and plotted these across NG → ES → LS, one
line per state. The effect of progression was summarised per state as the G4+
percentage-point change from NG to LS, accompanied by the NG-vs-LS odds ratio and
a two-by-two Fisher exact test (`scipy.stats.fisher_exact`) on
[G4+, G4−] × [NG, LS] counts. Because the unit of replication here is **promoters,
not donors**, p-values reflect table size rather than biological replication;
they are reported as a bounded value (clipped to `MIN_P = 1e-300`, with a
companion `log10_p_raw`) and the stats table explicitly flags
`unit = promoters (NOT donors)` (`fig3A2_g4_increase_<scheme>_stats.csv`).
Per-state counts and percentages are in `fig3A2_g4_increase_<scheme>_rawdata.csv`.
G4 accumulated preferentially at active and bivalent promoters across progression
(scheme `ac_me3`: Active 66.5 → 77.8 %, Bivalent 45.4 → 78.4 %, Repressed
8.8 → 11.1 %). Each scheme was rendered as counts, percentage and a combined
two-panel form.

### Chromatin-state flow split by G4 (`fig3_histone_state_flow.py`, Panel B)

Promoter chromatin state was tracked across NG → ES → LS as side-by-side
thick-stack alluvials split by G4. Chromatin state was defined from the
per-promoter histone marks under four schemes, each stated explicitly on the
panel:

- **`ac_me3`** (4 states): Active = H3K27ac; Bivalent = H3K27ac + H3K27me3;
  Repressed = H3K27me3; Other = no mark (H3K4me1 not used).
- **`poised_primed`** (6 states): adds Poised = H3K4me1 + H3K27me3 (no H3K27ac)
  and Primed = H3K4me1 only.
- **`k4me1_bivalent`** (4 states): H3K4me1 + H3K27me3 folds into Bivalent.
- **`single_primed`** (5 states): Primed = any H3K4me1 without H3K27ac.

Throughout the figure each state is labelled by its constituent histone
modification(s) in legends and definition lines (Active = H3K27ac, orange;
Bivalent = H3K27ac + H3K27me3, magenta; Poised = H3K4me1 + H3K27me3, green;
Primed = H3K4me1, purple; Repressed = H3K27me3, blue; Other = no mark, grey).

Note on the "Active" label across figures: at **promoters** (this figure) the
Active state is defined by H3K27ac, because promoter H3K4me1 is not informative
for promoter activity. At **distal enhancers** (Figure 4, and the enhancer
overlap in Figure 5 panel L1), the Active state requires H3K27ac **and** H3K4me1
(and not H3K27me3), reflecting the canonical active-enhancer signature. The same
word therefore denotes a promoter-specific (H3K27ac) and an enhancer-specific
(H3K27ac + H3K4me1) state; each figure's legend states the marks explicitly.

Each scheme was crossed with three expression sets (all promoters; genes with
TPM > 1 in any stage; genes with TPM > 0.1 in any stage) and with two G4-split
modes, giving 4 × 3 × 2 = 24 panels, each with a companion source-data table
(per-promoter marks, state per stage, G4 status and stage-mean TPM):

- **Static-presence split (`any`)**: G4+ = G4-positive in any stage (n = 16,093)
  vs G4− = never G4+ (n = 22,748).
- **Trajectory split (`gained`)**: a three-way, non-overlapping partition of all
  promoters into GAINED (G4− at NG and G4+ at ES or LS; n = 6,531), constitutive
  G4+ at NG (n = 9,562) and never G4+ (n = 22,748).

G4+ promoters were strongly enriched for the Active state relative to G4−
promoters (LS_EAC 78.3 % vs 10.8 %), and the Active fraction increased across
progression; the K4me1-aware schemes resolved a primed/poised pool otherwise
scored as Other. This panel is descriptive (flows and proportions); no
inferential test was applied to the flow.

### Gene-level over-representation of gained-G4 promoters (`fig3_go_gained.py`, Panel C)

We tested which gene functions are over-represented among promoters that **gain**
a G4 during progression. Query gene sets were defined from the corrected BG4 G4
status as genes whose promoter was G4− at NG and G4+ at the target stage, for
three transitions: NG→ES (G4+ at ES_EAC; n = 4,778), NG→LS (G4+ at LS_EAC;
n = 5,626) and NG→ES/LS (G4+ at ES_EAC or LS_EAC; n = 6,531).

Over-representation was assessed by Fisher exact test (two-sided,
`scipy.stats.fisher_exact`) on the 2×2 table of query-membership × gene-set
membership within a background, restricting to gene sets with ≥5 background genes
and ≥1 query gene. The effect size is the odds ratio (shown as log2 odds ratio).
Multiple testing across terms within each transition was corrected by
Benjamini–Hochberg FDR computed in a numerically stable, underflow-safe form
(raw p clipped to `MIN_P = 1e-300`, with companion `log10_p_raw` and
`log10_q_fdr`). Because the unit of replication is **genes, not donors**, large
query sizes inflate significance; results are read primarily on the odds ratio.

Four query/background versions were run, the choice of background being the
analytical point:

- **V1 — gained-G4 promoters vs all genes (n = 38,841) [PRIMARY].** The framing
  is chromatin architecture: where do G4 structures form in the genome during
  progression? The relevant question is the function of G4-gaining promoter
  regions versus the whole genome.
- **V2 — expressed gainers vs all genes.**
- **V3 — gained-G4 promoters vs the expressed-gene background.**
- **V4 — expressed gainers vs the expressed-gene background [control].** The
  expressed background is all genes with stage-mean TPM > 0.1 in any stage
  (n = 17,863); for V2/V4 the query is restricted to expressed gainers so it
  remains a subset of the background, as Fisher over-representation requires.

V3/V4 are an expression-matched robustness footnote (to pre-empt the objection
that the signal merely reflects that G4+ promoters are expressed), not the
headline; the enrichment attenuates but survives expression-matching. Each of the
5 ontologies (GO Biological Process 2025, KEGG 2026, Reactome 2024, MSigDB
Hallmark 2020, MSigDB Oncogenic [refs]) × 4 versions = 20 heatmaps. In each
heatmap, rows are the union of the top 15 FDR-significant enriched terms (q < 0.05,
direction enriched) per transition, ranked by the NG→ES log2 odds ratio; colour
encodes log2 odds ratio on an RdBu_r scale and asterisks encode FDR
(* q < 0.05, ** q < 0.01, *** q < 0.001, **** q < 1e-10). Per-heatmap source data
(every term × transition: overlap, odds ratio, raw/adjusted p with log10
companions, and the overlapping gene list) and a single consolidated table
merging all terms/transitions/versions (`fig3C_GO_master_results.csv`; 84,144
rows; columns method, ontology, background, transition, term, overlap, effect,
effect_type = odds_ratio, p_raw, q_fdr, log10_q_fdr, genes) were written.
G4-gain marked cancer-progression programmes (DNA repair, p53/RB, cell
cycle/proliferation, glycolysis, TGF-β/EMT, interferon/TNF inflammation) with
odds ratios of ~2–3× versus all genes that survived expression-matching.

> **A region-based (GREAT-style) test was prototyped and removed.** A local
> GREAT-style binomial regulatory-domain enrichment (after McLean et al., *GREAT
> improves functional interpretation of cis-regulatory regions*, Nat Biotechnol
> 28:495, 2010 [ref]) was implemented but **not used**: the binomial null assumes
> peaks could fall anywhere genome-wide, whereas the gained-G4 query is
> promoter-restricted, so peaks artificially over-hit gene regulatory domains and
> the in-house implementation returned implausible fold-enrichments (100–470×).
> The gene-level Fisher ORA above is the GO analysis we report; for a validated
> region-based analysis the official GREAT web tool should be used.

### Data-scaled colorbar heatmap variants (`fig3_heatmaps_tpm01.py`)

For the expressed-gainer analysis (query = genes with TPM > 0.1 in any stage; the
V2 `gainedExpr_vs_allgenes` background), the Panel C ontologies and the Panel D
cancer summary were re-rendered with the colorbar scaled to each heatmap's true
log2-odds-ratio range (rather than a fixed symmetric cap), with the actual
minimum/midpoint/maximum labelled at the colorbar (e.g. Hallmark 1.12–2.50; Panel
D 1.40–4.40). The RdBu_r colour map and the FDR star annotation are unchanged;
only the scale differs, so the modest expressed-gainer effects use the full
colour range. Implemented via the shared helper `config.effect_heatmap(...,
scale='data')`, which returns the image plus (vmin, vmax); `scale='symmetric'`
(the default centred at 0) leaves all other panels unchanged. Inputs are read
from `fig3C_GO_master_results.csv`; outputs include
`fig3C_GO_gained_TPM01_<ontology>.{png,svg}`,
`fig3D_cancer_hallmark_summary_TPM01.{png,svg}` and companion source-data tables.

### Curated cancer-programme summary (`fig3_cancer_summary.py`, Panel D)

A focused, cross-ontology heatmap presents 18 hand-curated, EAC-relevant cancer
programmes that gain promoter G4, drawn from the significant terms of the V1
primary analysis (`fig3C_GO_master_results.csv`, background
`V1_gained_vs_allgenes`, method ORA_Fisher) and grouped by hallmark (p53/RB, cell
cycle/proliferation, DNA repair, Warburg metabolism, EMT/invasion,
inflammation/immune, oncogenic signalling). Columns are the three gain transitions
(NG→ES, NG→LS, NG→ES/LS); colour is the log2 odds ratio (RdBu_r) and asterisks
the FDR tier. This is the reviewer-facing EAC narrative panel; Panel C is the
exhaustive supplement. Source data: `fig3D_cancer_hallmark_summary_rawdata.csv`.

### EAC driver-gene G4 status (`fig3_driver_panel.py`, Panel E)

The chromatin-architecture finding was mapped onto a curated list of 33 canonical
EAC/Barrett's driver genes (from TCGA-ESCA, Dulak et al. 2013, Frankell et al.
2019, Stachler et al. 2015 [refs]), each annotated as oncogene or tumour
suppressor. For each driver we plotted promoter G4 status at NG/ES_EAC/LS_EAC
(G4+ red, G4− blue), a 'gained-G4' flag (G4− at NG and G4+ at any later stage)
and a log2(TPM + 1) expression track from the corrected RNA-seq, with gene labels
coloured by oncogene (red) versus tumour suppressor (blue). Drivers were ordered
by gained / constitutive-G4+ / G4− status within class. 15 of 33 drivers gained a
promoter G4 during progression (including MYC, ERBB2, CDKN2A, MET, CDK6, MDM2,
RNF43, RB1, TGFBR2) and 14 were already G4+ at NG (including TP53, KRAS, SMAD4,
CCNE1, EGFR, PIK3CA). Source data: `fig3E_driver_g4_status_rawdata.csv`. This
panel is descriptive (per-gene categorical status); no inferential test was
applied.

### Timing of G4 gain: ES vs LS Venn and GO of timing-unique sets (`fig3_gained_ES_LS_venn_go.py`, Panel M)

Gained-G4 promoter sets (G4− at NG → G4+ at the stage) were compared between
early (ES_EAC) and late (LS_EAC) acquisition and shown as a Venn diagram
(`fig3M_gained_ES_LS_venn_rawdata.csv`). This partitioned gainers into shared
(gained early and retained), ES-only (gained at ES but lost by LS; transient) and
LS-only (late-acquired) sets. Over-representation of the ES-only and LS-only
unique gene sets was then tested separately against the all-genes background
(n = 38,841) by Fisher exact test with Benjamini–Hochberg FDR (same underflow-safe
implementation as Panel C: raw p clipped to 1e-300 with `log10_p_raw` and
`log10_q_fdr`), across the same 5 ontologies. As elsewhere, the unit of
replication is **promoters, not donors**; results are led by the effect size
(log2 odds ratio, displayed RdBu_r with FDR stars), with rows the union of the
top 15 FDR-significant enriched terms per set ranked by the ES-only log2 odds
ratio. Outputs include the per-ontology heatmaps
(`fig3M_GO_ESonly_LSonly_<ontology>.{png,svg}`), the timing-unique gene lists
(`fig3M_ES_only_genes.csv`, `fig3M_LS_only_genes.csv`) and a master GO table
(`fig3M_GO_ESonly_LSonly_master.csv`).

### Statistics and reporting conventions

Across Figure 3 the effect size is the primary readout: odds ratios (log2-scaled)
for over-representation and G4-state comparisons, and absolute percentage-point
changes for G4+ enrichment per chromatin state. Enrichment p-values are from
two-sided Fisher exact tests and corrected for multiple testing within each
analysis by Benjamini–Hochberg FDR computed in an underflow-safe manner: raw
p-values are clipped to a floor of 1e-300 and accompanied by a `log10_p_raw`
companion (and `log10_q_fdr` for the adjusted values), so the true magnitude is
preserved when SciPy underflows below the float64 floor and a literal p = 0 is
never reported. The unit of replication throughout is regions/promoters/genes,
**not donors**; because per-region tests read out table size rather than
biological replication, this pseudo-replication caveat is stated and the
effect-size-led interpretation is preferred over the nominal p-value.
G4-specificity claims are not GC-controlled, as GC content is intrinsic to G4
formation. Figures were saved as paired PNG (300 dpi) and SVG, with SVGs
post-processed for Affinity Designer 2 (editable Arial text, split CSS `font:`
shorthand, `<use>` repositioning, metadata stripped, `xmlns:xlink` ensured).

### Data and code

All inputs (canonical BG4 and histone peak sets, GENCODE v44 promoter
annotation, GMT ontology files, RNA-seq TPM matrix and corrected metadata) and
all outputs (per-panel `*_rawdata.csv` source-data tables, the consolidated
`fig3C_GO_master_results.csv`, and the panel figures as PNG + Affinity-clean SVG)
are contained in the self-contained `figure_3/` folder. Analyses are reproducible
via the scripts `fig3_derive_promoter_annotations.py` (per-promoter G4 + histone
annotation), `fig3_status_flow.py`, `fig3_g4_increase.py`,
`fig3_histone_state_flow.py`, `fig3_go_gained.py`, `fig3_heatmaps_tpm01.py`,
`fig3_cancer_summary.py`, `fig3_driver_panel.py` and
`fig3_gained_ES_LS_venn_go.py`, with shared configuration in `config.py`.
Software: Python 3.8.8, bedtools v2.26.0, NumPy, pandas, SciPy, Matplotlib,
seaborn.
