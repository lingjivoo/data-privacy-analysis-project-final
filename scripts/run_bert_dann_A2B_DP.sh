#!/bin/bash
# Train DP-DANN for domain adaptation A->B
# Uses train_domain_adapt_A2B_DP.py

set -e

echo "=========================================="
echo "BERT-DANN: Domain Adaptation A -> B"
echo "=========================================="

# Paths
A_TRAIN="corpus_A/A_train_leaky.jsonl"
A_VAL="corpus_A/A_val_leaky.jsonl"
B_TRAIN="corpus_B/B_train.jsonl"
B_VAL="corpus_B/B_val.jsonl"

# Pretrained BERT checkpoint (trained on A)
PRETRAINED_CKPT_DIR="ckpt_bert_A_finetune_max512"

# Check files exist
if [ ! -f "$A_TRAIN" ]; then
  echo "Error: $A_TRAIN not found"; exit 1
fi

if [ ! -f "$A_VAL" ]; then
  echo "Error: $A_VAL not found"; exit 1
fi

if [ ! -f "$B_TRAIN" ]; then
  echo "Error: $B_TRAIN not found"; exit 1
fi

if [ ! -d "$PRETRAINED_CKPT_DIR" ]; then
  echo "Error: pretrained checkpoint dir $PRETRAINED_CKPT_DIR not found"
  echo "Need to train baseline first on A (train_bert_nli_on_A.py, finetune_encoder mode)"
  echo "Or update PRETRAINED_CKPT_DIR to point to your checkpoint"
  exit 1
fi

echo "Using A-train: $A_TRAIN"
echo "Using A-val  : $A_VAL"
echo "Using B-train: $B_TRAIN"
echo "Pretrained ckpt dir: $PRETRAINED_CKPT_DIR"
echo ""

# Train DP-DANN with different epsilon values
echo ""
echo "[2/5] Training BERT-DANN with DP-SGD ..."

for EPS in 2.0 5.0 10.0; do
  OUT_DIR="checkpoints/ckpt_A2B_bert_dann_dp_eps${EPS}"
  echo "  -> Training DP-DANN with ε=${EPS}, out_dir=${OUT_DIR}"

  python scripts/train_domain_adapt_A2B_DP.py\
    --A_train_jsonl "$A_TRAIN" \
    --A_val_jsonl   "$A_VAL" \
    --B_train_jsonl "$B_TRAIN" \
    --B_val_jsonl   "$B_VAL" \
    --pretrained_ckpt_dir "$PRETRAINED_CKPT_DIR" \
    --out_dir "$OUT_DIR" \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --lambda_dom 0.1 \
    --dp_epsilon "$EPS" \
    --dp_delta 1e-5 \
    --dp_max_grad_norm 1.0 \
    --max_length 512 


done

echo ""
echo "=========================================="
echo "BERT-DANN domain adaptation finished!"
echo "Check ckpt_A2B_bert_dann* and results/dann_A2B_results.csv"
echo "=========================================="
