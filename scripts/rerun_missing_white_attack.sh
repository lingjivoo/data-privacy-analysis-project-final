#!/usr/bin/env bash
# 仅重新提交缺失结果的白盒攻击任务：
# - ckpt_A2B_bert_dann_dp_eps0.5 (B 数据)
# - ckpt_A2B_bert_dann_dp_eps10.0 (A 数据)
# - ckpt_A2B_bert_dann_dp_eps1.0 (A 数据)
set -euo pipefail

source /ibex/user/yangz0h/miniconda3/etc/profile.d/conda.sh
conda activate cs_225

ROOT="/home/yangz0h/courses/cs_325/data-privacy-analysis-project"
CHECKPOINTS_DIR="${ROOT}/checkpoints"
LOG_DIR="${ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
  echo "未找到检查点目录: ${CHECKPOINTS_DIR}" >&2
  exit 1
fi

# 目标列表：格式 ckpt_name:data_tag(A/B)
targets=(
  "ckpt_A2B_bert_dann_dp_eps0.05:B"
  "ckpt_A2B_bert_dann_dp_eps0.05:A"
  "ckpt_A2B_bert_dann_dp_eps0.1:B"
)

for entry in "${targets[@]}"; do
  ckpt_name="${entry%%:*}"
  dataset_name="${entry##*:}"  # A 或 B

  # 模型类型判定：默认 dann，特例 ckpt_bert_A_finetune_max512 用 baseline
  model_type="dann"
  if [[ "${ckpt_name}" == "ckpt_bert_A_finetune_max512" ]]; then
    model_type="baseline"
  fi

  if [[ "${dataset_name}" == "A" ]]; then
    data_tag="a"
    extra_args="--output_prefix white_box_attack_${ckpt_name}_a"
  else
    data_tag="b"
    extra_args="--train_jsonl corpus_B/B_train.jsonl --test_jsonl corpus_B/B_test.jsonl --output_prefix white_box_attack_${ckpt_name}_b"
  fi

  log_file="${LOG_DIR}/white_box_${ckpt_name}_${data_tag}.out"
  job_name="wbox_${ckpt_name}_${data_tag}"
  cmd="cd \"${ROOT}\" && python scripts/white_box_attack.py --num_steps 10 --ckpt_dir \"checkpoints/${ckpt_name}\" --model_type \"${model_type}\" ${extra_args}"

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

  echo "已重新提交: ${job_name} (数据集: ${dataset_name}, 模型类型: ${model_type}) -> JobID ${job_id}, 日志: ${log_file}"
done

echo "全部缺失任务已提交。如需汇总，请之后运行 scripts/summarize_white_attack.sh"

