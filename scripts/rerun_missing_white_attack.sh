#!/usr/bin/env bash
# Resubmit missing white-box attack jobs
set -euo pipefail

source /ibex/user/yangz0h/miniconda3/etc/profile.d/conda.sh
conda activate cs_225

ROOT="/home/yangz0h/courses/cs_325/data-privacy-analysis-project"
CHECKPOINTS_DIR="${ROOT}/checkpoints"
LOG_DIR="${ROOT}/slurm_logs"
mkdir -p "${LOG_DIR}"

if [[ ! -d "${CHECKPOINTS_DIR}" ]]; then
  echo "Checkpoint directory not found: ${CHECKPOINTS_DIR}" >&2
  exit 1
fi

# Jobs to resubmit: format ckpt_name:data_tag
targets=(
  "ckpt_A2B_bert_dann_dp_eps0.05:B"
  "ckpt_A2B_bert_dann_dp_eps0.05:A"
  "ckpt_A2B_bert_dann_dp_eps0.1:B"
)

for entry in "${targets[@]}"; do
  ckpt_name="${entry%%:*}"
  dataset_name="${entry##*:}"

  # Determine model type
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

  echo "Resubmitted: ${job_name} (dataset: ${dataset_name}, model type: ${model_type}) -> JobID ${job_id}, log: ${log_file}"
done

echo "All missing tasks submitted. Run scripts/summarize_white_attack.sh later to summarize"

