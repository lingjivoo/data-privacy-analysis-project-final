#!/usr/bin/env bash
# Batch submit white-box attack jobs to SLURM
set -euo pipefail

source /ibex/user/yangz0h/miniconda3/etc/profile.d/conda.sh
conda activate cs_225

ROOT="/home/yangz0h/courses/cs_325/data-privacy-analysis-project"
CHECKPOINTS_DIR="${ROOT}/checkpoints"
LOG_DIR="${ROOT}/slurm_logs"
SUMMARY_LOG="${LOG_DIR}/white_box_summary.log"

mkdir -p "${LOG_DIR}"

if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINTS_DIR}" >&2
  exit 1
fi

# Extract results from logs (confusion matrix etc)
extract_result_block() {
  local log_file="$1"
  # Get end of log, find confusion matrix section
  tail -n 200 "${log_file}" | awk '
    /混淆矩阵/ {flag=1}
    flag {print}
  '
}

declare -a submitted_logs=()

for ckpt_path in "${CHECKPOINTS_DIR}"/*; do
  [[ -d "${ckpt_path}" ]] || continue
  ckpt_name="$(basename "${ckpt_path}")"

  # Determine model type (default: dann, special case: baseline)
  model_type="dann"
  if [[ "${ckpt_name}" == "ckpt_bert_A_finetune_max512" ]]; then
    model_type="baseline"
  fi

  # Run on both datasets: a=A, b=B
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

    echo "Submitted task: ${job_name} (dataset: ${dataset_name}, model type: ${model_type}) -> JobID ${job_id}, log: ${log_file}"
  done
done
