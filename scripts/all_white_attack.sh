#!/usr/bin/env bash
# 批量对白盒攻击脚本提交 SLURM 任务，并汇总日志结果。
set -euo pipefail

source /ibex/user/yangz0h/miniconda3/etc/profile.d/conda.sh
conda activate cs_225

ROOT="/home/yangz0h/courses/cs_325/data-privacy-analysis-project"
CHECKPOINTS_DIR="${ROOT}/checkpoints"
LOG_DIR="${ROOT}/slurm_logs"
SUMMARY_LOG="${LOG_DIR}/white_box_summary.log"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
  echo "未找到检查点目录: ${CHECKPOINTS_DIR}" >&2
  exit 1
fi

# 将混淆矩阵等最终输出从日志中摘取出来
extract_result_block() {
  local log_file="$1"
  # 取日志末尾一部分，找到“混淆矩阵”开始的块
  tail -n 200 "${log_file}" | awk '
    /混淆矩阵/ {flag=1}
    flag {print}
  '
}

declare -a submitted_logs=()

for ckpt_path in "${CHECKPOINTS_DIR}"/*; do
  [[ -d "${ckpt_path}" ]] || continue
  ckpt_name="$(basename "${ckpt_path}")"

  # 模型类型判定：默认 dann，特例 ckpt_bert_A_finetune_max512 用 baseline
  model_type="dann"
  if [[ "${ckpt_name}" == "ckpt_bert_A_finetune_max512" ]]; then
    model_type="baseline"
  fi

  # 两份数据集：a 使用默认（A），b 使用 B 语料
  for data_tag in a b; do
    if [[ "${data_tag}" == "a" ]]; then
      extra_args="--output_prefix white_box_attack_${ckpt_name}_a"
      dataset_name="A"
    else
      extra_args="--train_jsonl corpus_B/B_train.jsonl --test_jsonl corpus_B/B_test.jsonl --output_prefix white_box_attack_${ckpt_name}_b"
      dataset_name="B"
    fi

    log_file="${LOG_DIR}/white_box_${ckpt_name}_${data_tag}.out"
    submitted_logs+=("${log_file}")

    job_name="wbox_${ckpt_name}_${data_tag}"
    cmd="cd \"${ROOT}\" && python scripts/white_box_attack.py --num_steps 5 --epochs 100 --train_size 2000 --ckpt_dir \"checkpoints/${ckpt_name}\" --model_type \"${model_type}\" ${extra_args}"

    job_id=$(sbatch \
      --time=03:30:00 \
      --nodes=1 \
      --gpus-per-node=v100:1 \
      --cpus-per-gpu=1 \
      --mem=32G \
      --partition=batch \
      --job-name="${job_name}" \
      --output="${log_file}" \
      --parsable \
      --wrap "${cmd}")

    echo "提交任务: ${job_name} (数据集: ${dataset_name}, 模型类型: ${model_type}) -> JobID ${job_id}, 日志: ${log_file}"
  done
done
