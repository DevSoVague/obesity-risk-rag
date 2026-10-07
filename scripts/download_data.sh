#!/usr/bin/env bash
# Download the public datasets used by this project. Nothing here is committed to git.
#
#   1. UCI "Estimation of obesity levels based on eating habits and physical condition"
#      -> app/ObesityDataSet_raw_and_data_sinthetic.csv (Model 1 training, model_pipeline.py)
#         and notebooks/ (00_model1_development.ipynb)
#   2. NHANES August 2021 - August 2023 (suffix _L) XPT files
#      -> notebooks/{demographics_data,exam_data,lab_data,quest_data}/ (notebooks 01-03)
#
# Usage: bash scripts/download_data.sh   (run from the repo root)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NB="$ROOT/notebooks"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== UCI obesity dataset"
UCI_URL="https://archive.ics.uci.edu/static/public/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip"
curl -fsSL "$UCI_URL" -o "$TMP/uci.zip"
unzip -o -q "$TMP/uci.zip" -d "$TMP/uci"
CSV="$(find "$TMP/uci" -name 'ObesityDataSet_raw_and_data_sinthetic.csv' | head -n 1)"
cp "$CSV" "$ROOT/app/"
cp "$CSV" "$NB/"

echo "== NHANES 2021-2023"
BASE="https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles"
fetch() {  # fetch <subfolder> <FILE ...>
  local dir="$NB/$1"; shift
  mkdir -p "$dir"
  for f in "$@"; do
    echo "   $f.xpt -> notebooks/$(basename "$dir")/"
    curl -fsSL "$BASE/$f.xpt" -o "$dir/$f.xpt"
  done
}
fetch demographics_data DEMO_L
fetch exam_data         BMX_L BPXO_L BAX_L
fetch lab_data          BIOPRO_L GHB_L GLU_L INS_L TCHOL_L
fetch quest_data        ALQ_L BAQ_L BPQ_L DIQ_L HSQ_L KIQ_U_L MCQ_L PAQ_L SLQ_L SMQ_L WHQ_L

echo "Done. Next: run notebooks/01-03 from inside notebooks/, then copy"
echo "notebooks/merged_data/ow_*fasting_clean.csv into app/merged_data/ if you want to run app/resave_bundles.py."
