#!/bin/bash -l
# Run mia_inference.py on all checkpoints for both A and B data
# Submit with: sbatch scripts/run_mia_inference_all.sh

#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=32G
#SBATCH --partition=batch
#SBATCH --job-name=mia_inf_all
#SBATCH --output=slurm_logs/mia_inf_all-%j.out
#SBATCH --error=slurm_logs/mia_inf_all-%j.err

set -euo pipefail

PROJECT_ROOT="/home/yangz0h/courses/cs_325/data-privacy-analysis-project"
CHECKPOINT_DIR="${PROJECT_ROOT}/checkpoints"
RESULT_DIR="${PROJECT_ROOT}/mia_inference_result"
mkdir -p "${RESULT_DIR}" "${PROJECT_ROOT}/slurm_logs"

source /ibex/user/yangz0h/miniconda3/etc/profile.d/conda.sh
conda activate cs_225

cd "${PROJECT_ROOT}"

for CKPT_PATH in "${CHECKPOINT_DIR}"/*; do
  [[ -d "${CKPT_PATH}" ]] || continue
  CKPT_NAME="$(basename "${CKPT_PATH}")"

  # Determine model type
  MODEL_TYPE="dann"
  if [[ "${CKPT_NAME}" == "ckpt_bert_A_finetune_max512" ]]; then
    MODEL_TYPE="baseline"
  fi

  for DATA_TAG in A B; do
    if [[ "${DATA_TAG}" == "A" ]]; then
      TRAIN_JSONL="${PROJECT_ROOT}/corpus_A/A_train_leaky.jsonl"
      TEST_JSONL="${PROJECT_ROOT}/corpus_A/A_test_leaky.jsonl"
    else
      TRAIN_JSONL="${PROJECT_ROOT}/corpus_B/B_train.jsonl"
      TEST_JSONL="${PROJECT_ROOT}/corpus_B/B_test.jsonl"
    fi

    if [[ ! -f "${TRAIN_JSONL}" || ! -f "${TEST_JSONL}" ]]; then
      echo "[skip] ${CKPT_NAME} ${DATA_TAG}: Data files missing, skipping"
      continue
    fi

    OUT_CSV="${RESULT_DIR}/${CKPT_NAME}_${DATA_TAG}.csv"

    echo "[run] Model=${CKPT_NAME} (type=${MODEL_TYPE}) Data=${DATA_TAG} -> ${OUT_CSV}"
    python scripts/mia_inference.py \
      --ckpt_dir "${CKPT_PATH}" \
      --model_type "${MODEL_TYPE}" \
      --train_jsonl "${TRAIN_JSONL}" \
      --test_jsonl "${TEST_JSONL}" \
      --output_csv "${OUT_CSV}"
  done
done

echo "[done] All models have been processed"

