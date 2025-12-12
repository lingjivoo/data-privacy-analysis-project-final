#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Membership Inference Attack (MIA) based on correct label probability and cross-entropy loss.

Read data from member_inference_records.csv, extract probability corresponding to gold_label for each sample:
  - gold_label=0 (entailment) -> p_entailment
  - gold_label=1 (contradiction) -> p_contradiction
  - gold_label=2 (neutral) -> p_neutral

Two attack methods:
1. Based on probability: samples above threshold are considered members, below are non-members
2. Based on cross-entropy: samples below threshold are considered members, above are non-members (low loss = member)

Plot attack accuracy under different thresholds.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Numerical stability to avoid log(0)
EPSILON = 1e-10


def load_records(csv_path: Path):
    """Load CSV records, return arrays of member labels, correct label probabilities, cross-entropy loss, Mentr, correct."""
    member_labels = []
    y_probs = []  # Probability corresponding to gold_label
    cross_entropy = []  # Cross-entropy loss: -log(p_correct)
    mentr = []  # Mentr values
    correct = []  # Whether prediction is correct

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            member_labels.append(int(row["member_label"]))
            gold_label = int(row["gold_label"])
            # Extract corresponding probability based on gold_label
            if gold_label == 0:  # entailment
                p_correct = float(row["p_entailment"])
            elif gold_label == 1:  # contradiction
                p_correct = float(row["p_contradiction"])
            elif gold_label == 2:  # neutral
                p_correct = float(row["p_neutral"])
            else:
                raise ValueError(f"Unknown gold_label: {gold_label}")
            
            y_probs.append(p_correct)
            # Compute cross-entropy loss: -log(p_correct), add small value to avoid log(0)
            ce_loss = -np.log(max(p_correct, EPSILON))
            cross_entropy.append(ce_loss)
            
            # Read Mentr value (if exists)
            mentr_value = float(row.get("mentr", 0.0))
            mentr.append(mentr_value)
            
            # Read whether prediction is correct
            correct_value = int(row.get("correct", 0))
            correct.append(correct_value)

    return np.array(member_labels), np.array(y_probs), np.array(cross_entropy), np.array(mentr), np.array(correct)


def search_best_threshold(member_labels, y_probs, num_thresholds=1000):
    """
    Search for best threshold (based on probability), return:
      - thresholds: all attempted thresholds
      - accuracies: attack accuracy corresponding to each threshold
      - best_threshold: best threshold
      - best_accuracy: best accuracy
    """
    min_prob = y_probs.min()
    max_prob = y_probs.max()
    thresholds = np.linspace(min_prob, max_prob, num_thresholds)
    accuracies = []

    for thresh in thresholds:
        # Samples above threshold are considered members (1), below are non-members (0)
        attack_pred = (y_probs > thresh).astype(int)
        # Compute attack accuracy: match between predicted member/non-member and true labels
        # Formula: accuracy = (number of correctly predicted samples) / (total samples)
        # Numerator = TP + TN = (true member and predicted as member) + (true non-member and predicted as non-member)
        # Denominator = TP + TN + FP + FN = total samples
        acc = accuracy_score(member_labels, attack_pred)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    # Find threshold with maximum attack accuracy (i.e., best attack performance)
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    return thresholds, accuracies, best_threshold, best_accuracy


def search_best_threshold_ce(member_labels, cross_entropy, num_thresholds=1000):
    """
    Search for best threshold (based on cross-entropy), return:
      - thresholds: all attempted thresholds
      - accuracies: attack accuracy corresponding to each threshold
      - best_threshold: best threshold
      - best_accuracy: best accuracy
    
    Note: Lower cross-entropy loss indicates higher likelihood of being a member (model memorizes well, low loss)
    """
    min_ce = cross_entropy.min()
    max_ce = cross_entropy.max()
    thresholds = np.linspace(min_ce, max_ce, num_thresholds)
    accuracies = []

    for thresh in thresholds:
        # Samples below threshold are considered members (1), above are non-members (0)
        attack_pred = (cross_entropy < thresh).astype(int)
        # Compute attack accuracy: match between predicted member/non-member and true labels
        # Formula: accuracy = (number of correctly predicted samples) / (total samples)
        # Numerator = TP + TN = (true member and predicted as member) + (true non-member and predicted as non-member)
        # Denominator = TP + TN + FP + FN = total samples
        acc = accuracy_score(member_labels, attack_pred)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    # Find threshold with maximum attack accuracy (i.e., best attack performance)
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    return thresholds, accuracies, best_threshold, best_accuracy


def search_best_threshold_mentr(member_labels, mentr, num_thresholds=1000):
    """
    Search for best threshold (based on Mentr), return:
      - thresholds: all attempted thresholds
      - accuracies: attack accuracy corresponding to each threshold
      - best_threshold: best threshold
      - best_accuracy: best accuracy
    
    Note: Lower Mentr values indicate higher likelihood of being a member (model is more confident on member samples)
    """
    min_mentr = mentr.min()
    max_mentr = mentr.max()
    thresholds = np.linspace(min_mentr, max_mentr, num_thresholds)
    accuracies = []

    for thresh in thresholds:
        # Samples below threshold are considered members (1), above are non-members (0)
        attack_pred = (mentr < thresh).astype(int)
        # Compute attack accuracy: match between predicted member/non-member and true labels
        # Formula: accuracy = (number of correctly predicted samples) / (total samples)
        # Numerator = TP + TN = (true member and predicted as member) + (true non-member and predicted as non-member)
        # Denominator = TP + TN + FP + FN = total samples
        acc = accuracy_score(member_labels, attack_pred)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    # Find threshold with maximum attack accuracy (i.e., best attack performance)
    best_idx = np.argmax(accuracies)
    best_threshold = thresholds[best_idx]
    best_accuracy = accuracies[best_idx]

    return thresholds, accuracies, best_threshold, best_accuracy


def compute_attack_metrics(member_labels, attack_pred):
    """Compute attack Accuracy/Precision/Recall/F1 and confusion matrix."""
    acc = accuracy_score(member_labels, attack_pred)
    prec = precision_score(member_labels, attack_pred)
    rec = recall_score(member_labels, attack_pred)
    f1 = f1_score(member_labels, attack_pred)
    cm = confusion_matrix(member_labels, attack_pred)
    return acc, prec, rec, f1, cm


def evaluate_correct_as_member(member_labels, correct):
    """
    Method 4: Correct prediction is considered member, incorrect prediction is considered non-member
    This is a simple binary classification, no threshold search needed
    
    Returns:
      - accuracy: attack accuracy
      - attack_pred: attack prediction results
    """
    # Correct prediction = member (1), incorrect prediction = non-member (0)
    attack_pred = correct.astype(int)
    accuracy = accuracy_score(member_labels, attack_pred)
    
    # For interface consistency, return dummy threshold and accuracy arrays
    # Actually this method only has one value
    thresholds = np.array([0.0, 1.0])  # Dummy thresholds
    accuracies = np.array([accuracy, accuracy])  # Same accuracy
    
    return thresholds, accuracies, 0.5, accuracy, attack_pred


def plot_threshold_search(thresholds, accuracies, best_threshold, best_accuracy, out_path: Path, method_name="Probability"):
    """Plot threshold search process."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, accuracies, linewidth=1.5, label="Attack Accuracy", color="#2ca02c")
    plt.axvline(
        best_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Best Threshold = {best_threshold:.4f}",
    )
    plt.axhline(
        best_accuracy,
        color="red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label=f"Best Accuracy = {best_accuracy:.4f}",
    )
    if method_name == "Probability":
        plt.xlabel("Threshold (Probability of Correct Label)")
    elif method_name == "Cross-Entropy":
        plt.xlabel("Threshold (Cross-Entropy Loss)")
    else:  # Mentr
        plt.xlabel("Threshold (Mentr)")
    plt.ylabel("Attack Accuracy")
    plt.title(f"Member Inference Attack: {method_name} Threshold Search\nBest Threshold = {best_threshold:.4f}, Best Accuracy = {best_accuracy:.4f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[mia_y_prob] {method_name} threshold search plot saved: {out_path}")


def plot_distribution_comparison(member_labels, y_probs, cross_entropy, mentr, out_path: Path):
    """Plot comparison of train set (members) and test set (non-members) distributions on three metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    train_mask = member_labels == 1
    test_mask = member_labels == 0
    
    train_probs = y_probs[train_mask]
    test_probs = y_probs[test_mask]
    train_ce = cross_entropy[train_mask]
    test_ce = cross_entropy[test_mask]
    train_mentr = mentr[train_mask]
    test_mentr = mentr[test_mask]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Left plot: Softmax probability distribution of correct class
    ax = axes[0]
    ax.hist(train_probs, bins=50, alpha=0.6, label=f"Train (member, n={len(train_probs)})", 
            color="#1f77b4", density=True, range=(0.96, 1.0))
    ax.hist(test_probs, bins=50, alpha=0.6, label=f"Test (non-member, n={len(test_probs)})", 
            color="#ff7f0e", density=True, range=(0.96, 1.0))
    ax.set_xlabel("Probability of Correct Label")
    ax.set_ylabel("Density")
    ax.set_xlim(0.96, 1.0)
    ax.set_title(f"Distribution: Probability of Correct Label\nTrain mean={train_probs.mean():.4f}, Test mean={test_probs.mean():.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Middle plot: Cross-entropy loss distribution
    ax = axes[1]
    ax.hist(train_ce, bins=50, alpha=0.6, label=f"Train (member, n={len(train_ce)})", 
            color="#1f77b4", density=True, range=(0, 0.05))
    ax.hist(test_ce, bins=50, alpha=0.6, label=f"Test (non-member, n={len(test_ce)})", 
            color="#ff7f0e", density=True, range=(0, 0.05))
    ax.set_xlabel("Cross-Entropy Loss")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 0.05)
    ax.set_title(f"Distribution: Cross-Entropy Loss\nTrain mean={train_ce.mean():.4f}, Test mean={test_ce.mean():.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right plot: Mentr distribution
    ax = axes[2]
    ax.hist(train_mentr, bins=50, alpha=0.6, label=f"Train (member, n={len(train_mentr)})", 
            color="#1f77b4", density=True, range=(0, 0.001))
    ax.hist(test_mentr, bins=50, alpha=0.6, label=f"Test (non-member, n={len(test_mentr)})", 
            color="#ff7f0e", density=True, range=(0, 0.001))
    ax.set_xlabel("Mentr")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 0.001)
    ax.set_title(f"Distribution: Mentr\nTrain mean={train_mentr.mean():.4f}, Test mean={test_mentr.mean():.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[mia_y_prob] Distribution comparison plot saved: {out_path}")


def plot_comparison(prob_results, ce_results, mentr_results, correct_results, out_path: Path):
    """Plot comparison of threshold search results for four methods."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    prob_thresholds, prob_accuracies, prob_best_thresh, prob_best_acc = prob_results
    ce_thresholds, ce_accuracies, ce_best_thresh, ce_best_acc = ce_results
    mentr_thresholds, mentr_accuracies, mentr_best_thresh, mentr_best_acc = mentr_results
    correct_thresholds, correct_accuracies, correct_best_thresh, correct_best_acc = correct_results
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Probability method
    ax = axes[0, 0]
    ax.plot(prob_thresholds, prob_accuracies, linewidth=1.5, label="Attack Accuracy", color="#2ca02c")
    ax.axvline(prob_best_thresh, color="red", linestyle="--", linewidth=1.5, label=f"Best = {prob_best_thresh:.4f}")
    ax.axhline(prob_best_acc, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Threshold (Probability)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(f"Method 1: Probability\nBest Accuracy = {prob_best_acc:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top right: Cross-entropy method
    ax = axes[0, 1]
    ax.plot(ce_thresholds, ce_accuracies, linewidth=1.5, label="Attack Accuracy", color="#9467bd")
    ax.axvline(ce_best_thresh, color="red", linestyle="--", linewidth=1.5, label=f"Best = {ce_best_thresh:.4f}")
    ax.axhline(ce_best_acc, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Threshold (Cross-Entropy Loss)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(f"Method 2: Cross-Entropy\nBest Accuracy = {ce_best_acc:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom left: Mentr method
    ax = axes[1, 0]
    ax.plot(mentr_thresholds, mentr_accuracies, linewidth=1.5, label="Attack Accuracy", color="#d62728")
    ax.axvline(mentr_best_thresh, color="red", linestyle="--", linewidth=1.5, label=f"Best = {mentr_best_thresh:.4f}")
    ax.axhline(mentr_best_acc, color="red", linestyle="--", linewidth=1.5, alpha=0.5)
    ax.set_xlabel("Threshold (Mentr)")
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(f"Method 3: Mentr\nBest Accuracy = {mentr_best_acc:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Bottom right: Correct prediction method (no threshold search needed, directly show accuracy)
    ax = axes[1, 1]
    # This method doesn't need threshold search, so draw a simple bar chart
    ax.bar([0], [correct_best_acc], width=0.5, color="#ff7f0e", alpha=0.7, edgecolor="black")
    ax.axhline(correct_best_acc, color="red", linestyle="--", linewidth=1.5, alpha=0.5, label=f"Accuracy = {correct_best_acc:.4f}")
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0, 1])
    ax.set_xticks([])
    ax.set_ylabel("Attack Accuracy")
    ax.set_title(f"Method 4: Correct as Member\nAccuracy = {correct_best_acc:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[mia_y_prob] Comparison plot saved: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_path",
        default="results/member_inference_records.csv",
        help="Input CSV file path",
    )
    ap.add_argument(
        "--num_thresholds",
        type=int,
        default=1000,
        help="Number of thresholds to search (default 1000)",
    )
    ap.add_argument(
        "--output_prefix",
        default="mia_y_prob",
        help="Output file prefix (will generate results/{prefix}_*.png/json)",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"[mia_y_prob] Loading data: {csv_path}")
    member_labels, y_probs, cross_entropy, mentr, correct = load_records(csv_path)
    print(f"[mia_y_prob] Number of samples: {len(member_labels)}")
    print(f"[mia_y_prob] Member samples: {member_labels.sum()}")
    print(f"[mia_y_prob] Non-member samples: {len(member_labels) - member_labels.sum()}")
    print(f"[mia_y_prob] Correct label probability range: [{y_probs.min():.4f}, {y_probs.max():.4f}]")
    print(f"[mia_y_prob] Cross-entropy loss range: [{cross_entropy.min():.4f}, {cross_entropy.max():.4f}]")
    print(f"[mia_y_prob] Mentr range: [{mentr.min():.4f}, {mentr.max():.4f}]")
    print(f"[mia_y_prob] Correctly predicted samples: {correct.sum()}, Incorrectly predicted samples: {len(correct) - correct.sum()}")

    # Method 1: Threshold search based on probability
    print(f"\n[mia_y_prob] Method 1: Threshold search based on probability ({args.num_thresholds} thresholds)...")
    prob_thresholds, prob_accuracies, prob_best_threshold, prob_best_accuracy = search_best_threshold(
        member_labels, y_probs, args.num_thresholds
    )
    prob_attack_pred = (y_probs > prob_best_threshold).astype(int)
    prob_acc, prob_prec, prob_rec, prob_f1, prob_cm = compute_attack_metrics(member_labels, prob_attack_pred)
    print(f"[mia_y_prob] Probability method - Best threshold: {prob_best_threshold:.6f}")
    print(f"[mia_y_prob] Probability method - Acc: {prob_acc:.4f}, Member Recall: {prob_rec:.4f}, Member F1: {prob_f1:.4f}")

    # Method 2: Threshold search based on cross-entropy
    print(f"\n[mia_y_prob] Method 2: Threshold search based on cross-entropy ({args.num_thresholds} thresholds)...")
    ce_thresholds, ce_accuracies, ce_best_threshold, ce_best_accuracy = search_best_threshold_ce(
        member_labels, cross_entropy, args.num_thresholds
    )
    ce_attack_pred = (cross_entropy < ce_best_threshold).astype(int)
    ce_acc, ce_prec, ce_rec, ce_f1, ce_cm = compute_attack_metrics(member_labels, ce_attack_pred)
    print(f"[mia_y_prob] Cross-entropy method - Best threshold: {ce_best_threshold:.6f}")
    print(f"[mia_y_prob] Cross-entropy method - Acc: {ce_acc:.4f}, Member Recall: {ce_rec:.4f}, Member F1: {ce_f1:.4f}")
    
    # Method 3: Threshold search based on Mentr
    print(f"\n[mia_y_prob] Method 3: Threshold search based on Mentr ({args.num_thresholds} thresholds)...")
    mentr_thresholds, mentr_accuracies, mentr_best_threshold, mentr_best_accuracy = search_best_threshold_mentr(
        member_labels, mentr, args.num_thresholds
    )
    mentr_attack_pred = (mentr < mentr_best_threshold).astype(int)
    mentr_acc, mentr_prec, mentr_rec, mentr_f1, mentr_cm = compute_attack_metrics(member_labels, mentr_attack_pred)
    print(f"[mia_y_prob] Mentr method - Best threshold: {mentr_best_threshold:.6f}")
    print(f"[mia_y_prob] Mentr method - Acc: {mentr_acc:.4f}, Member Recall: {mentr_rec:.4f}, Member F1: {mentr_f1:.4f}")
    
    # Method 4: Correct prediction as member
    print(f"\n[mia_y_prob] Method 4: Correct prediction as member (no threshold search needed)...")
    correct_thresholds, correct_accuracies, correct_best_threshold, correct_best_accuracy, correct_attack_pred = evaluate_correct_as_member(
        member_labels, correct
    )
    correct_acc, correct_precision, correct_recall, correct_f1, correct_cm = compute_attack_metrics(member_labels, correct_attack_pred)
    print(f"[mia_y_prob] Correct prediction method - Acc: {correct_acc:.4f}, Member Recall: {correct_recall:.4f}, Member F1: {correct_f1:.4f}")
    print(f"[mia_y_prob] Correct prediction method - Confusion matrix: TN={correct_cm[0,0]}, FP={correct_cm[0,1]}, FN={correct_cm[1,0]}, TP={correct_cm[1,1]}")

    # Distribution comparison plot (train vs test)
    distribution_path = Path("results") / f"{args.output_prefix}_distribution_comparison.png"
    print(f"\n[mia_y_prob] Generating distribution comparison plot...")
    plot_distribution_comparison(member_labels, y_probs, cross_entropy, mentr, distribution_path)

    # Save results
    summary = {
        "method_1_probability": {
            "best_threshold": float(prob_best_threshold),
            "best_accuracy": float(prob_acc),
            "precision": float(prob_prec),
            "recall": float(prob_rec),
            "f1_score": float(prob_f1),
            "confusion_matrix": {
                "tn": int(prob_cm[0, 0]),
                "fp": int(prob_cm[0, 1]),
                "fn": int(prob_cm[1, 0]),
                "tp": int(prob_cm[1, 1]),
            },
        },
        "method_2_cross_entropy": {
            "best_threshold": float(ce_best_threshold),
            "best_accuracy": float(ce_acc),
            "precision": float(ce_prec),
            "recall": float(ce_rec),
            "f1_score": float(ce_f1),
            "confusion_matrix": {
                "tn": int(ce_cm[0, 0]),
                "fp": int(ce_cm[0, 1]),
                "fn": int(ce_cm[1, 0]),
                "tp": int(ce_cm[1, 1]),
            },
        },
        "method_3_mentr": {
            "best_threshold": float(mentr_best_threshold),
            "best_accuracy": float(mentr_acc),
            "precision": float(mentr_prec),
            "recall": float(mentr_rec),
            "f1_score": float(mentr_f1),
            "confusion_matrix": {
                "tn": int(mentr_cm[0, 0]),
                "fp": int(mentr_cm[0, 1]),
                "fn": int(mentr_cm[1, 0]),
                "tp": int(mentr_cm[1, 1]),
            },
        },
        "method_4_correct_as_member": {
            "best_threshold": float(correct_best_threshold),
            "best_accuracy": float(correct_best_accuracy),
            "precision": float(correct_precision),
            "recall": float(correct_recall),
            "f1_score": float(correct_f1),
            "confusion_matrix": {
                "tn": int(correct_cm[0, 0]),
                "fp": int(correct_cm[0, 1]),
                "fn": int(correct_cm[1, 0]),
                "tp": int(correct_cm[1, 1]),
            },
        },
        "distribution_comparison_plot_path": str(distribution_path),
        "num_samples": int(len(member_labels)),
        "num_members": int(member_labels.sum()),
        "num_non_members": int(len(member_labels) - member_labels.sum()),
        "y_prob_min": float(y_probs.min()),
        "y_prob_max": float(y_probs.max()),
        "y_prob_mean": float(y_probs.mean()),
        "y_prob_std": float(y_probs.std()),
        "ce_min": float(cross_entropy.min()),
        "ce_max": float(cross_entropy.max()),
        "ce_mean": float(cross_entropy.mean()),
        "ce_std": float(cross_entropy.std()),
        "mentr_min": float(mentr.min()),
        "mentr_max": float(mentr.max()),
        "mentr_mean": float(mentr.mean()),
        "mentr_std": float(mentr.std()),
        "csv_path": str(csv_path),
    }

    summary_path = Path("results") / f"{args.output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[mia_y_prob] Summary saved: {summary_path}")

    # Save detailed threshold search data (optional)
    prob_detail_path = Path("results") / f"{args.output_prefix}_prob_thresholds.csv"
    with open(prob_detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "accuracy"])
        for thresh, acc in zip(prob_thresholds, prob_accuracies):
            writer.writerow([f"{thresh:.6f}", f"{acc:.6f}"])
    print(f"[mia_y_prob] Probability method detailed threshold data saved: {prob_detail_path}")
    
    ce_detail_path = Path("results") / f"{args.output_prefix}_ce_thresholds.csv"
    with open(ce_detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "accuracy"])
        for thresh, acc in zip(ce_thresholds, ce_accuracies):
            writer.writerow([f"{thresh:.6f}", f"{acc:.6f}"])
    print(f"[mia_y_prob] Cross-entropy method detailed threshold data saved: {ce_detail_path}")
    
    mentr_detail_path = Path("results") / f"{args.output_prefix}_mentr_thresholds.csv"
    with open(mentr_detail_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "accuracy"])
        for thresh, acc in zip(mentr_thresholds, mentr_accuracies):
            writer.writerow([f"{thresh:.6f}", f"{acc:.6f}"])
    print(f"[mia_y_prob] Mentr method detailed threshold data saved: {mentr_detail_path}")


if __name__ == "__main__":
    main()

