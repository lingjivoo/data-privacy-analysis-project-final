#!/bin/bash -l
# Process all CSVs in mia_inference_result with mia_y_prob.py
# Outputs go to results/y_prob_results/
# Usage: bash scripts/run_mia_y_prob_all.sh

set -euo pipefail

source /ibex/user/yangz0h/miniconda3/etc/profile.d/conda.sh
conda activate cs_225

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CSV_DIR="${PROJECT_ROOT}/mia_inference_result"
RESULT_PREFIX_DIR="${PROJECT_ROOT}/results/y_prob_results"
LOG_FILE="${RESULT_PREFIX_DIR}/all.log"
mkdir -p "${RESULT_PREFIX_DIR}"

cd "${PROJECT_ROOT}"

echo "===== run_mia_y_prob_all $(date '+%F %T') =====" > "${LOG_FILE}"

for csv_file in "${CSV_DIR}"/*.csv; do
  [[ -f "${csv_file}" ]] || continue
  base_name="$(basename "${csv_file}" .csv)"
  output_prefix="y_prob_results/${base_name}"
  echo "[run] ${base_name} -> ${output_prefix}" | tee -a "${LOG_FILE}"
  python scripts/mia_y_prob.py \
    --csv_path "${csv_file}" \
    --output_prefix "${output_prefix}" | tee -a "${LOG_FILE}"
done

echo "[done] All CSVs processed, results located at results/y_prob_results/" | tee -a "${LOG_FILE}"

