#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script based on A_train_leaky.jsonl (members) and A_test_leaky.jsonl (non-members).
Process:
  1) Reuse eval_on_B's model loading logic, run model to get logits/softmax.
  2) Record softmax, predicted labels, whether prediction is correct, etc. for each sample.
  3) True member labels: train=1, test=0.
  4) Output member_inference_records.csv.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 同目录下的推理工具
from eval_on_B import (
    LABEL2ID,
    TextDatasetXY,
    build_model_and_tokenizer,
    load_jsonl_xy,
    make_collate,
)

# Numerical stability to avoid log(0)
EPSILON = 1e-10


def compute_mentr(probs, y_true):
    """
    Compute Mentr(F(x), y) = -(1 - F(x)_y) log(F(x)_y) - Σ_{i ≠ y} F(x)_i log(1 - F(x)_i)
    
    Args:
        probs: numpy array of shape [batch_size, num_classes] - probability distribution
        y_true: numpy array of shape [batch_size] - true labels
    
    Returns:
        mentr_values: numpy array of shape [batch_size] - Mentr value for each sample
    """
    batch_size, num_classes = probs.shape
    mentr_values = np.zeros(batch_size)
    
    for i in range(batch_size):
        p_correct = probs[i, y_true[i]]
        # First term: -(1 - F(x)_y) log(F(x)_y)
        term1 = -(1 - p_correct) * np.log(max(p_correct, EPSILON))
        
        # Second term: -Σ_{i ≠ y} F(x)_i log(1 - F(x)_i)
        term2 = 0.0
        for j in range(num_classes):
            if j != y_true[i]:
                p_incorrect = probs[i, j]
                # Avoid log(1 - p) issues when p=1
                p_incorrect = min(p_incorrect, 1.0 - EPSILON)
                term2 -= p_incorrect * np.log(1 - p_incorrect)
        
        mentr_values[i] = term1 + term2
    
    return mentr_values


def infer_dataset(model, tokenizer, X_txt, y, max_length, device, model_type, batch_size=64):
    """Perform inference on given text collection, return softmax/predictions/labels."""
    ds = TextDatasetXY(X_txt, y)
    collate = make_collate(tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.eval()
    all_probs, all_preds, all_labels, all_mentr = [], [], [], []

    with torch.no_grad():
        for input_ids, attn_mask, yb in dl:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            yb = yb.to(device)

            if model_type == "baseline":
                logits = model(input_ids, attn_mask)
            else:  # dann
                logits = model(input_ids, attn_mask,
                               lambd=0.0, return_domain=False)

            # Prediction method exactly consistent with eval_on_B.py
            pred = logits.argmax(dim=1)
            
            # Compute softmax probabilities for membership inference analysis (not needed in eval_on_B.py)
            # logits are unnormalized scores (can be positive or negative), need softmax to convert to probability distribution
            probs = torch.softmax(logits, dim=1)
            probs_np = probs.cpu().numpy()
            yb_np = yb.cpu().numpy()
            
            # Compute Mentr values
            mentr_batch = compute_mentr(probs_np, yb_np)
            
            all_probs.append(probs_np)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb_np)
            all_mentr.extend(mentr_batch)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_mentr = np.array(all_mentr)
    return all_probs, all_preds, all_labels, all_mentr


def save_csv(rows, out_path: Path):
    """Save inference results to CSV file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[mia_correct] 结果已写入 {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", help="Directory containing encoder_meta.json and weights",
                    default="checkpoints/ckpt_bert_A_finetune_max512")
    ap.add_argument("--model_type", choices=["baseline", "dann"], default="baseline")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument(
        "--train_jsonl",
        default="corpus_A/A_train_leaky.jsonl",
        help="Member samples (train)",
    )
    ap.add_argument(
        "--test_jsonl",
        default="corpus_A/A_test_leaky.jsonl",
        help="Non-member samples (test)",
    )
    ap.add_argument(
        "--output_csv",
        default="results/member_inference_records.csv",
        help="Output CSV file path (default: results/member_inference_records.csv)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    # Ensure path parsing is correct: if relative path, base on parent directory of script directory (project root)
    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.is_absolute():
        # Get parent directory of script directory (project root)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        ckpt_dir = project_root / ckpt_dir
    
    # Check if path exists
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    print(f"[mia_correct] Using checkpoint directory: {ckpt_dir.absolute()}")

    print(f"[mia_correct] Loading model and tokenizer...")
    tokenizer, model, max_length, _ = build_model_and_tokenizer(
        ckpt_dir, args.model_type, device
    )

    # Load data (load_jsonl_xy will handle relative paths)
    train_jsonl = Path(args.train_jsonl)
    test_jsonl = Path(args.test_jsonl)
    if not train_jsonl.is_absolute():
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        train_jsonl = project_root / train_jsonl
        test_jsonl = project_root / test_jsonl
    
    print(f"[mia_correct] Loading training data: {train_jsonl}")
    X_train, y_train = load_jsonl_xy(str(train_jsonl))
    print(f"[mia_correct] Loading test data: {test_jsonl}")
    X_test, y_test = load_jsonl_xy(str(test_jsonl))

    print(f"[mia_correct] Inferring on training set ({len(X_train)} samples)...")
    probs_tr, preds_tr, labels_tr, mentr_tr = infer_dataset(
        model, tokenizer, X_train, y_train, max_length, device, args.model_type, args.batch_size
    )
    print(f"[mia_correct] Inferring on test set ({len(X_test)} samples)...")
    probs_te, preds_te, labels_te, mentr_te = infer_dataset(
        model, tokenizer, X_test, y_test, max_length, device, args.model_type, args.batch_size
    )

    # Assemble member/non-member labels: train=1, test=0
    member_labels = np.concatenate(
        [np.ones(len(labels_tr), dtype=int), np.zeros(len(labels_te), dtype=int)]
    )
    all_probs = np.concatenate([probs_tr, probs_te], axis=0)
    all_preds = np.concatenate([preds_tr, preds_te], axis=0)
    all_labels = np.concatenate([labels_tr, labels_te], axis=0)
    all_mentr = np.concatenate([mentr_tr, mentr_te], axis=0)
    # max_softmax: maximum softmax probability across all classes for each sample
    # i.e., max(p_entailment, p_contradiction, p_neutral) for each sample
    max_softmax = all_probs.max(axis=1)

    print(f"[mia_correct] Writing results to CSV...")
    # Write CSV: record line by line
    rows = []
    for idx, (probs, pred, gold, mem, mentr) in enumerate(
        zip(all_probs, all_preds, all_labels, member_labels, all_mentr)
    ):
        rows.append(
            {
                "idx": idx,
                "member_label": int(mem),
                "gold_label": int(gold),
                "pred_label": int(pred),
                "correct": int(pred == gold),
                "max_softmax": float(probs.max()),
                "mentr": float(mentr),
                "p_entailment": float(probs[LABEL2ID["entailment"]]),
                "p_contradiction": float(probs[LABEL2ID["contradiction"]]),
                "p_neutral": float(probs[LABEL2ID["neutral"]]),
            }
        )

    # Parse output path
    csv_path = Path(args.output_csv)
    if not csv_path.is_absolute():
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        csv_path = project_root / csv_path
    
    save_csv(rows, csv_path)
    print(f"[mia_correct] Done! Total samples: {len(rows)} (members: {member_labels.sum()}, non-members: {len(member_labels) - member_labels.sum()})")


if __name__ == "__main__":
    main()

