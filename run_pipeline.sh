#!/bin/bash
# DP-MedNLI Complete Experimental Pipeline Script
# Executes all steps according to README Phase I-III
# Requirements: scripts/train_bert_nli_on_A.py and other scripts are ready

set -e  # Exit immediately on error

echo "=========================================="
echo "DP-MedNLI: Complete Pipeline"
echo "=========================================="

# ------------------------------------------
# Phase I: Data Construction
# ------------------------------------------
echo ""
echo "Phase I: Privacy-Aware Data Construction"
echo "----------------------------------------"

# 1. Standardize data (if needed)
if [ -f "corpus_A/train.txt" ]; then
    echo "[1/4] Converting train.txt to JSONL..."
    python scripts/convert_to_standard_jsonl.py \
        --mode train_txt \
        --input corpus_A/train.txt \
        --output corpus_A/A_train_leaky.jsonl
fi

# 2. Inject synthetic PHI (add "leak labels" on domain A)
if [ -f "corpus_A/A_train_leaky.jsonl" ]; then
    echo "[2/4] Injecting synthetic PHI into Hospital A data..."
    python scripts/make_synthetic_phi_A.py \
        --in_jsonl corpus_A/A_train_leaky.jsonl \
        --out_jsonl corpus_A/A_train_leaky.jsonl \
        --leak_prob 0.35 \
        --hospital_name "Hospital A"
fi


# 3. Generate Hospital B data (A->B mapping)
echo "[3/4] Generating Hospital B data from Hospital A..."
for split in train val test; do
    if [ -f "corpus_A/A_${split}_leaky.jsonl" ]; then
        python scripts//hospitalB_shift_pipeline.py \
            --in_jsonl corpus_A/A_${split}_leaky.jsonl \
            --out_jsonl corpus_B/B_${split}.jsonl \
            --config corpus_B/shift_config.json
    fi
done


echo "Phase I completed!"

# ------------------------------------------
# Phase II: Baseline Training (BERT, with/without DP)
# ------------------------------------------
echo ""
echo "Phase II: Baseline & DP Training on A (BERT)"
echo "--------------------------------------------"

# Naming convention:
#   - No DP: ckpt_bert_A_finetune
#   - With DP: ckpt_bert_A_finetune_dp_eps${eps}

# [1/5] Train BERT baseline (no DP, finetune encoder)
echo "[1/5] Training BERT baseline (no DP, finetune encoder)..."
python scripts/train_bert_nli_on_A.py \
      --train_jsonl corpus_A/A_train_leaky.jsonl \
      --val_jsonl corpus_A/A_val_leaky.jsonl  \
      --out_dir ckpt_bert_A_finetune \
      --finetune_encoder  \
      --batch_size 32  \
      --epochs 20

# [2-5/5] Train BERT with DP (multiple epsilon values)
for eps in 0.5 1.0 2.0 8.0; do
    echo "[2-5/5] Training BERT with DP (ε=${eps})..."
    python scripts/train_bert_nli_on_A.py \
      --train_jsonl corpus_A/A_train_leaky.jsonl \
      --val_jsonl corpus_A/A_val_leaky.jsonl  \
      --out_dir ckpt_bert_A_finetune_dp_eps${eps}  \
      --finetune_encoder  \
      --batch_size 8  \
      --epochs 10 \
      --dp_epsilon ${eps} \
      --dp_delta 1e-5 \
      --dp_max_grad_norm 1.0
done

echo "Phase II completed!"

# ------------------------------------------
# Phase III: Domain Adaptation (DANN / DP-DANN)
# ------------------------------------------
echo ""
echo "Phase III: Domain Adaptation (DANN & DP-DANN)"
echo "----------------------------------------------"

# Naming:
#   - No DP: ckpt_A2B_dann
#   - With DP: ckpt_A2B_dp_eps${eps}

# [1/5] DANN (no DP)
echo "[1/5] Training DANN (no DP)..."
python scripts/train_domain_adapt_A2B.py \
    --A_train_jsonl corpus_A/A_train_leaky.jsonl \
    --A_val_jsonl corpus_A/A_val_leaky.jsonl \
    --B_train_jsonl corpus_B/B_train.jsonl \
    --B_val_jsonl corpus_B/B_val.jsonl \
    --pretrained_ckpt_dir ckpt_bert_A_finetune \
    --out_dir ckpt_A2B_dann \
    --epochs 10 \
    --lambda_dom 0.1 \
    --lr 1e-4 \

# [2-5/5] DP-DANN (multiple epsilon values)
for eps in 0.01 0.05 0.1 0.5 1.0 2.0 4.0 8.0; do
    echo "[2-5/5] Training BERT with DP (ε=${eps})..."
    python scripts/train_domain_adapt_A2B_DP.py \
      --A_train_jsonl corpus_A/A_train_leaky.jsonl \
      --A_val_jsonl corpus_A/A_val_leaky.jsonl  \
      --B_val_jsonl corpus_B/B_val.jsonl \
      --pretrained_ckpt_dir ckpt_bert_A_finetune
      --out_dir ckpt_bert_A_finetune_dp_eps${eps}  \
      --epochs 10 \
      --dp_epsilon ${eps} \
      --dp_delta 1e-5 \
      --dp_max_grad_norm 1.0 \
      --lambda_dom 0.1 \
      --lr 1e-4 \
done


echo "Phase III completed!"

# ------------------------------------------
# Evaluation: Utility on B
# ------------------------------------------
echo ""
echo "Evaluation: Utility on Hospital B"
echo "----------------------------------"

# Evaluate all BERT baseline models (including DP)
for ckpt in ckpt_bert_A_finetune \
            ckpt_bert_A_finetune_dp_eps0.5 \
            ckpt_bert_A_finetune_dp_eps1.0 \
            ckpt_bert_A_finetune_dp_eps2.0 \
            ckpt_bert_A_finetune_dp_eps8.0; do
    if [ -d "${ckpt}" ]; then
        echo "Evaluating baseline (BERT) on B: ${ckpt}..."
        python scripts/eval_on_B.py \
            --B_test_jsonl corpus_B/B_test.clean.jsonl \
            --ckpt_dir ${ckpt} \
            --model_type baseline
    fi
done

# Evaluate all DANN models (including DP-DANN)
for ckpt in ckpt_A2B_dann \
            ckpt_A2B_dp_eps0.5 \
            ckpt_A2B_dp_eps1.0 \
            ckpt_A2B_dp_eps2.0 \
            ckpt_A2B_dp_eps8.0; do
    if [ -d "${ckpt}" ]; then
        echo "Evaluating DANN on B: ${ckpt}..."
        python scripts/eval_on_B.py \
            --B_test_jsonl corpus_B/B_test.clean.jsonl \
            --ckpt_dir ${ckpt} \
            --model_type dann
    fi
done

echo "Utility evaluation completed!"

# ------------------------------------------
# Evaluation: Privacy Attacks
# ------------------------------------------
echo ""
echo "Evaluation: Privacy Attacks"
echo "---------------------------"

# MIA attack (requires A_train + A_test)
if [ -f "corpus_A/A_train_leaky.jsonl" ] && [ -f "corpus_A/A_test_leaky.jsonl" ]; then
    # BERT baseline models
    for ckpt in ckpt_bert_A_finetune \
                ckpt_bert_A_finetune_dp_eps0.5 \
                ckpt_bert_A_finetune_dp_eps1.0 \
                ckpt_bert_A_finetune_dp_eps2.0 \
                ckpt_bert_A_finetune_dp_eps8.0; do
        if [ -d "${ckpt}" ]; then
            echo "MIA attack on baseline (BERT) model: ${ckpt}..."
            python scripts/attack_membership_inference.py \
                --ckpt_dir ${ckpt} \
                --model_type baseline \
                --train_jsonl corpus_A/A_train_leaky.jsonl \
                --test_jsonl corpus_A/A_test_leaky.jsonl
        fi
    done

    # DANN / DP-DANN
    for ckpt in ckpt_A2B_dann \
                ckpt_A2B_dp_eps0.5 \
                ckpt_A2B_dp_eps1.0 \
                ckpt_A2B_dp_eps2.0 \
                ckpt_A2B_dp_eps8.0; do
        if [ -d "${ckpt}" ]; then
            echo "MIA attack on DANN model: ${ckpt}..."
            python scripts/attack_membership_inference.py \
                --ckpt_dir ${ckpt} \
                --model_type dann \
                --train_jsonl corpus_A/A_train_leaky.jsonl \
                --test_jsonl corpus_A/A_test_leaky.jsonl
        fi
    done
fi

# PHI scanning (on B_test)
echo "Scanning for PHI-like patterns on B_test..."
if [ -f "corpus_B/B_test.clean.jsonl" ]; then
    python scripts/scan_phi_like.py corpus_B/B_test.clean.jsonl
fi

echo "Privacy evaluation completed!"

# ------------------------------------------
# Generate Plots
# ------------------------------------------
echo ""
echo "Generating plots..."
python scripts/plot_tradeoff.py --summary

echo ""
echo "=========================================="
echo "Pipeline completed successfully!"
echo "Check results/ directory for outputs"
echo "=========================================="
