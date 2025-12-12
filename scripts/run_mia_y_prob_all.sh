#!/bin/bash -l
# 遍历 mia_inference_result 下的所有 CSV，依次运行 mia_y_prob.py
# 结果统一写到 results/y_prob_results/ 前缀下
# 用法：bash scripts/run_mia_y_prob_all.sh

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
  # 输出前缀：results/y_prob_results/<base_name>_y_prob
  output_prefix="y_prob_results/${base_name}"
  echo "[run] ${base_name} -> ${output_prefix}" | tee -a "${LOG_FILE}"
  python scripts/mia_y_prob.py \
    --csv_path "${csv_file}" \
    --output_prefix "${output_prefix}" | tee -a "${LOG_FILE}"
done

echo "[done] 所有 CSV 已处理，结果位于 results/y_prob_results/" | tee -a "${LOG_FILE}"

