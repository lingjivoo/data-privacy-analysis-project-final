#!/usr/bin/env bash
# 汇总 slurm_logs 中白盒攻击的输出结果（混淆矩阵等末尾块）
set -euo pipefail

ROOT="/home/yangz0h/courses/cs_325/data-privacy-analysis-project"
LOG_DIR="${ROOT}/slurm_logs"
SUMMARY_LOG="${LOG_DIR}/white_box_summary.log"

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "未找到日志目录: ${LOG_DIR}" >&2
  exit 1
fi

# 提取每个日志中“混淆矩阵”开始的末尾块
extract_result_block() {
  local log_file="$1"
  tail -n 200 "${log_file}" | awk '
    /最终测试准确率/ {flag=1}
    flag {
      print
      # 如果遇到保存文件的提示，停止提取（混淆矩阵之后的内容）
      if (/训练曲线已保存|ROC 曲线已保存|结果摘要已保存/) {
        exit
      }
    }
  '
}

log_files=( "${LOG_DIR}"/white_box_*.out )

if [[ ${#log_files[@]} -eq 1 && ! -e "${log_files[0]}" ]]; then
  echo "未找到 white_box_*.out 日志文件，请先运行提交脚本。" >&2
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
      echo "(日志存在，但尚未找到包含最终测试准确率的结果块；任务可能尚未结束)"
    fi
  done
} > "${SUMMARY_LOG}"

echo "汇总完成: ${SUMMARY_LOG}"
