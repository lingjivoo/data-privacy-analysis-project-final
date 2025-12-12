#!/bin/bash
# Train BERT-DANN for domain adaptation A->B
# Uses train_domain_adapt_A2B.py

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

# Train DANN (no DP)
echo "[1/5] Training BERT-DANN (no DP)..."


python scripts/train_domain_adapt_A2B.py \
  --A_train_jsonl "$A_TRAIN" \
  --A_val_jsonl   "$A_VAL" \
  --B_train_jsonl "$B_TRAIN" \
  --B_val_jsonl   "$B_VAL" \
  --pretrained_ckpt_dir "$PRETRAINED_CKPT_DIR" \
  --out_dir ckpt_A2B_bert_dann_new \
  --epochs 20 \
  --batch_size 8  \
  --lr 2e-5 \
  --lambda_dom 0.1 \
  --early_stop_patience 3


echo ""
echo "=========================================="
echo "BERT-DANN domain adaptation finished!"
echo "Check ckpt_A2B_bert_dann* and results/dann_A2B_results.csv"
echo "=========================================="
