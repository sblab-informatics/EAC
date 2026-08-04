#!/usr/bin/env bash
# build_chromatin_consensus.sh
# Rebuild per-condition consensus peak sets for histone marks + ATAC from the
# RAW per-replicate SEACR/peak BEDs, on the MANUAL cell-line classification and
# the project-canonical reproducibility rule:
#
#   - Per donor: its single technical replicate (histone & ATAC have 1 rep/donor,
#     T1 only -> no 2-of-3 technical step is possible; for BG4, which has 3 reps,
#     the 2-of-3 step was applied upstream).
#   - Per condition: a region is kept if present in >= 2 of the stage's DONORS
#     (>=2-of-N). Single-donor stages (NG, ES_Rx, LS_Rx) cannot apply >=2-of-N and
#     use that donor's own peaks, FLAGGED n=1 (not reproducibility-filtered).
#   - Canonical human chroms only (chr1-22, X, Y); chrM / E.coli / contigs dropped.
#
# Manual classification (filename label -> analytical stage; NOT trusting filename):
#   AHM1678/NG->NG  CAM450/NG->LS_EAC  CAM408,CAM412,WTSI_OESO_146/ES->ES_EAC
#   WTSI_OESO_117/LS->ES_EAC  CAM277,CAM574/LS->LS_EAC  CAM428/ES_Rx->ES_Rx
#   CAM629/LS_Rx->LS_Rx
set -euo pipefail

HIST_RAW=/scratche/slow/sblab/melidi01/Nader/Figures_monday_26J/bed_files/histone_marks
ATAC_RAW=/scratche/slow/sblab/melidi01/Nader/Figures_monday_26J/bed_files/ATAC_seq
OUT=/mnt/scratche/slow/sblab/melidi01/Nader/g4_chromatin_standalone/Publication/pre_publication_11Jun/_raw_data/chromatin_consensus_by_manual_stage
mkdir -p "$OUT"
SHM=/dev/shm/${USER}_chromcons; mkdir -p "$SHM"
trap 'rm -rf "$SHM"' EXIT
CANON='^chr([1-9]|1[0-9]|2[0-2]|X|Y)\b'

# stage -> "donor:filecond donor:filecond ..."  (manual classification)
declare -A STAGE_DONORS=(
  [NG]="AHM1678:NG"
  [ES_EAC]="CAM408:ES CAM412:ES WTSI_OESO_146:ES WTSI_OESO_117:LS"
  [LS_EAC]="CAM277:LS CAM574:LS CAM450:NG"
  [ES_Rx]="CAM428:ES_Rx"
  [LS_Rx]="CAM629:LS_Rx"
)
STAGES=(NG ES_EAC LS_EAC ES_Rx LS_Rx)

# find a donor's raw per-rep BED for a given assay; echo path or empty
find_bed() {
  local assay=$1 donor=$2 cond=$3
  if [[ "$assay" == "ATAC" ]]; then
    ls "$ATAC_RAW"/*organoid."$donor"."$cond".ATAC*.bed 2>/dev/null | head -1
  else
    ls "$HIST_RAW"/*organoid."$donor"."$cond"."$assay".*.bed 2>/dev/null | head -1
  fi
}

# clean a raw bed -> 3-col, canonical chroms, sorted, merged (per-donor set)
donor_set() {
  local raw=$1 out=$2
  grep -E "$CANON" "$raw" | cut -f1-3 | sort -k1,1 -k2,2n | bedtools merge -i - > "$out"
}

echo "assay        stage     donors                         n_peaks  rule"
for assay in H3K27ac H3K4me1 H3K27me3 ATAC; do
  for stage in "${STAGES[@]}"; do
    donors=(${STAGE_DONORS[$stage]})
    donor_beds=()
    names=()
    for dc in "${donors[@]}"; do
      donor=${dc%%:*}; cond=${dc##*:}
      raw=$(find_bed "$assay" "$donor" "$cond")
      if [[ -z "$raw" ]]; then echo "  WARN missing $assay $donor $cond"; continue; fi
      ds="$SHM/${assay}_${stage}_${donor}.bed"
      donor_set "$raw" "$ds"
      donor_beds+=("$ds"); names+=("$donor")
    done
    n=${#donor_beds[@]}
    out="$OUT/${assay}_${stage}.bed"
    if (( n == 0 )); then echo "  WARN no donors for $assay $stage"; continue; fi
    if (( n == 1 )); then
      # single-donor stage: that donor's own peaks (flagged n=1)
      awk -v s="$stage" -v a="$assay" 'BEGIN{OFS="\t"}{print $1,$2,$3,a"_"s"_"NR}' "${donor_beds[0]}" > "$out"
      rule="n=1 (single donor)"
    else
      # >=2-of-N donors: multiinter donor support >=2, then merge
      bedtools multiinter -i "${donor_beds[@]}" | awk '$4>=2' | cut -f1-3 \
        | sort -k1,1 -k2,2n | bedtools merge -i - \
        | awk -v s="$stage" -v a="$assay" 'BEGIN{OFS="\t"}{print $1,$2,$3,a"_"s"_"NR}' > "$out"
      rule=">=2/$n donors"
    fi
    np=$(wc -l < "$out")
    printf "%-12s %-8s %-30s %-8s %s\n" "$assay" "$stage" "$(IFS=,;echo "${names[*]}")" "$np" "$rule"
  done
done

echo ""
echo "Done. Consensus BEDs in: $OUT"
