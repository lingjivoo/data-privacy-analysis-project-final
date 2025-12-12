#!/usr/bin/env bash
# Summarize white-box attack results from slurm_logs
set -euo pipefail

ROOT="/home/yangz0h/courses/cs_325/data-privacy-analysis-project"
LOG_DIR="${ROOT}/slurm_logs"
SUMMARY_LOG="${LOG_DIR}/white_box_summary.log"

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "Log directory not found: ${LOG_DIR}" >&2
  exit 1
fi

# Extract result block from each log
extract_result_block() {
  local log_file="$1"
  tail -n 200 "${log_file}" | awk '
    /最终测试准确率/ {flag=1}
    flag {
      print
      # Stop when we hit file save messages
      if (/训练曲线已保存|ROC 曲线已保存|结果摘要已保存/) {
        exit
      }
    }
  '
}

log_files=( "${LOG_DIR}"/white_box_*.out )

if [[ ${#log_files[@]} -eq 1 && ! -e "${log_files[0]}" ]]; then
  echo "No white_box_*.out log files found, please run submission script first." >&2
  exit 1
fi

{
  echo "===== White-box attack summary $(date '+%Y-%m-%d %H:%M:%S') ====="
  for log_file in "${log_files[@]}"; do
    [[ -f "${log_file}" ]] || continue
    log_title="$(basename "${log_file}")"
    echo ""
    echo "### ${log_title}"
    result_block="$(extract_result_block "${log_file}")"
    if [[ -n "${result_block}" ]]; then
      echo "${result_block}"
    else
      echo "(Log exists but result block containing final test accuracy not found yet; task may not have finished)"
    fi
  done
} > "${SUMMARY_LOG}"

echo "Summary completed: ${SUMMARY_LOG}"
