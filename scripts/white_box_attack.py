#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved white-box membership inference attack (White-box MIA).

Main improvements:
1. No longer restore parameters, let the model truly learn
2. Add more differentiating features (initial loss, loss decrease rate, etc.)
3. Use multiple sample batches to extract features
4. Add higher-order features such as gradient variance, Hessian approximation
5. Compare model behavior differences before and after training
"""

import argparse
import csv
import json
from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from eval_on_B import (
    LABEL2ID,
    TextDatasetXY,
    build_model_and_tokenizer,
    load_jsonl_xy,
    make_collate,
)


class WhiteBoxFeatureDataset(Dataset):
    """White-box feature dataset"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class WhiteBoxClassifier(nn.Module):
    """Classifier for white-box attack"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.4):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def compute_sample_wise_gradient(model, input_ids, attn_mask, yb, criterion, model_type):
    """Compute sample-wise gradients (for calculating gradient variance)"""
    model.zero_grad()
    
        # Compute gradients sample by sample
        sample_grads = []
        for i in range(input_ids.size(0)):
            model.zero_grad()
            
            inp = input_ids[i:i+1]
            att = attn_mask[i:i+1]
            y = yb[i:i+1]
            
            # Forward pass
            if model_type == "dann":
                outputs = model(inp, att, lambd=0.0, return_domain=False)
                # DANN may return a tuple, take the first element
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
            else:
                logits = model(inp, att)
            
            loss = criterion(logits, y)
            loss.backward()
            
            # Collect gradients
            grads = []
            for p in model.parameters():
                if p.grad is not None:
                    grads.append(p.grad.view(-1).detach().cpu().numpy())
            
            if grads:
                sample_grads.append(np.concatenate(grads))
    
    return sample_grads


def extract_white_box_features(model, tokenizer, X_txt, y, max_length, device, 
                               model_type, num_steps=5, batch_size=8, lr=0.001):
    """
    Improved white-box feature extraction.
    
    Main improvements:
    1. Use larger batch_size and more appropriate learning rate
    2. Do not restore parameters, let the model truly learn
    3. Extract richer features (initial state, training trajectory, gradient statistics, etc.)
    4. Add higher-order features such as inter-sample gradient variance
    """
    
    # Save original model state (for comparison)
    original_model = copy.deepcopy(model)
    original_model.eval()
    
    model.train()
    
    ds = TextDatasetXY(X_txt, y)
    collate = make_collate(tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    criterion = nn.CrossEntropyLoss(reduction='none')  # Use 'none' to get per-sample loss
    criterion_mean = nn.CrossEntropyLoss()
    
    all_features = []
    
    print(f"[white_box] Starting white-box feature extraction, samples: {len(X_txt)}, batch_size: {batch_size}, training steps: {num_steps}")
    
    # First compute features for all samples in initial state (before training)
    print(f"[white_box] Step 1/3: Computing initial state features...")
    initial_features = []
    with torch.no_grad():
        for input_ids, attn_mask, yb in dl:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            yb = yb.to(device)
            
            # Use original model
            if model_type == "baseline":
                logits = original_model(input_ids, attn_mask)
            else:
                logits = original_model(input_ids, attn_mask, lambd=0.0, return_domain=False)
            
            losses = criterion(logits, yb).cpu().numpy()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            
            for i in range(len(losses)):
                p = probs[i]
                sorted_p = np.sort(p)[::-1]
                initial_features.append({
                    'initial_loss': losses[i],
                    'initial_entropy': -np.sum(p * np.log(p + 1e-12)),
                    'initial_max_prob': sorted_p[0],
                    'initial_top2_gap': sorted_p[0] - (sorted_p[1] if len(sorted_p) > 1 else 0.0),
                    'initial_prob_std': p.std(),
                    'initial_correct': int(np.argmax(p) == yb[i].item()),
                })
    
    # Train and extract trajectory features
    print(f"[white_box] Step 2/3: Training model and extracting trajectory features...")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    trajectory_features = [[] for _ in range(len(X_txt))]  # Training trajectory for each sample
    sample_idx_offset = 0
    
    for step in range(num_steps):
        step_losses = []
        step_grad_norms = []
        step_grad_vars = []
        
        for batch_idx, (input_ids, attn_mask, yb) in enumerate(dl):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if model_type == "baseline":
                logits = model(input_ids, attn_mask)
            else:
                logits = model(input_ids, attn_mask, lambd=0.0, return_domain=False)
            
            # Compute loss for each sample
            losses = criterion(logits, yb)
            loss_mean = losses.mean()
            
            # Backward pass
            loss_mean.backward()
            
            # Compute gradient statistics
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_grad_norm += param_norm ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            # Compute inter-sample gradient variance (calculate every 3 steps to save time)
            if step % 3 == 0 and input_ids.size(0) > 1:
                sample_grads = compute_sample_wise_gradient(
                    model, input_ids, attn_mask, yb, criterion_mean, model_type
                )
                if len(sample_grads) > 1:
                    grad_var = np.var(sample_grads, axis=0).mean()
                else:
                    grad_var = 0.0
            else:
                grad_var = 0.0
            
            # Update parameters
            optimizer.step()
            
            # Forward pass again to get updated state
            with torch.no_grad():
                if model_type == "baseline":
                    logits_after = model(input_ids, attn_mask)
                else:
                    logits_after = model(input_ids, attn_mask, lambd=0.0, return_domain=False)
                
                probs_after = torch.softmax(logits_after, dim=1).cpu().numpy()
            
            # Record features for each sample
            losses_np = losses.detach().cpu().numpy()
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            
            for i in range(len(losses_np)):
                sample_idx = sample_idx_offset + i
                
                p_before = probs[i]
                p_after = probs_after[i]
                
                trajectory_features[sample_idx].append({
                    'loss': losses_np[i],
                    'grad_norm': total_grad_norm / len(losses_np),  # Average gradient norm
                    'grad_var': grad_var,
                    'entropy_before': -np.sum(p_before * np.log(p_before + 1e-12)),
                    'entropy_after': -np.sum(p_after * np.log(p_after + 1e-12)),
                    'max_prob_before': p_before.max(),
                    'max_prob_after': p_after.max(),
                    'prob_change': np.abs(p_after - p_before).sum(),
                })
            
            sample_idx_offset += len(losses_np)
        
        # Reset offset for next epoch
        sample_idx_offset = 0
        
        if (step + 1) % 2 == 0:
            avg_loss = np.mean([t[-1]['loss'] for t in trajectory_features if len(t) > 0])
            print(f"  Step {step+1}/{num_steps} - Average Loss: {avg_loss:.4f}")
    
    # Build final feature vectors
    print(f"[white_box] Step 3/3: Building feature vectors...")
    for sample_idx in range(len(X_txt)):
        feature_vec = []
        
        # 1. Initial state features (6 dimensions)
        init_feat = initial_features[sample_idx]
        feature_vec.extend([
            init_feat['initial_loss'],
            init_feat['initial_entropy'],
            init_feat['initial_max_prob'],
            init_feat['initial_top2_gap'],
            init_feat['initial_prob_std'],
            init_feat['initial_correct'],
        ])
        
        # 2. Training trajectory features
        traj = trajectory_features[sample_idx]
        if len(traj) > 0:
            # Loss at each step (num_steps dimensions)
            losses = [t['loss'] for t in traj]
            feature_vec.extend(losses)
            
            # Gradient norm at each step (num_steps dimensions)
            grad_norms = [t['grad_norm'] for t in traj]
            feature_vec.extend(grad_norms)
            
            # Loss statistics (7 dimensions)
            feature_vec.extend([
                np.mean(losses),
                np.std(losses),
                np.min(losses),
                np.max(losses),
                losses[-1],
                losses[0] - losses[-1],  # Loss decrease amount
                (losses[0] - losses[-1]) / (losses[0] + 1e-8),  # Loss decrease rate
            ])
            
            # Gradient statistics (5 dimensions)
            feature_vec.extend([
                np.mean(grad_norms),
                np.std(grad_norms),
                np.min(grad_norms),
                np.max(grad_norms),
                grad_norms[-1],
            ])
            
            # Gradient variance statistics (3 dimensions)
            grad_vars = [t['grad_var'] for t in traj]
            feature_vec.extend([
                np.mean(grad_vars),
                np.max(grad_vars),
                np.std(grad_vars),
            ])
            
            # Entropy changes (4 dimensions)
            entropy_changes = [traj[i]['entropy_after'] - traj[i]['entropy_before'] 
                             for i in range(len(traj))]
            feature_vec.extend([
                np.mean(entropy_changes),
                np.std(entropy_changes),
                np.min(entropy_changes),
                np.max(entropy_changes),
            ])
            
            # Probability changes (4 dimensions)
            prob_changes = [t['prob_change'] for t in traj]
            feature_vec.extend([
                np.mean(prob_changes),
                np.std(prob_changes),
                np.min(prob_changes),
                np.max(prob_changes),
            ])
            
            # Max prob changes (2 dimensions)
            max_prob_changes = [traj[i]['max_prob_after'] - traj[i]['max_prob_before']
                               for i in range(len(traj))]
            feature_vec.extend([
                np.mean(max_prob_changes),
                np.sum(max_prob_changes),
            ])
            
        else:
            # If no trajectory, pad with zeros
            expected_dim = num_steps * 2 + 7 + 5 + 3 + 4 + 4 + 2
            feature_vec.extend([0.0] * expected_dim)
        
        all_features.append(feature_vec)
        
        if (sample_idx + 1) % 100 == 0:
            print(f"  Processed {sample_idx + 1}/{len(X_txt)} samples")
    
    features_array = np.array(all_features)
    print(f"[white_box] Feature extraction completed! Feature dimension: {features_array.shape[1]}")
    
    return features_array


def load_original_data(train_jsonl: Path, test_jsonl: Path):
    """Load text and labels from original JSONL files"""
    X_train, y_train = load_jsonl_xy(str(train_jsonl))
    X_test, y_test = load_jsonl_xy(str(test_jsonl))
    
    X_all = X_train + X_test
    y_all = list(y_train) + list(y_test)
    
    return X_all, y_all


def train_white_box_classifier(train_features, train_labels, test_features, test_labels,
                               device, epochs=300, batch_size=64, lr=0.001):
    """Train white-box attack classifier"""
    
    # Data standardization
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # 创建数据集
    train_dataset = WhiteBoxFeatureDataset(train_features_scaled, train_labels)
    test_dataset = WhiteBoxFeatureDataset(test_features_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_dim = train_features.shape[1]
    model = WhiteBoxClassifier(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        num_classes=2,
        dropout=0.4
    )
    model.to(device)
    
    print(f"[white_box] Classifier input feature dimension: {input_dim}")
    print(f"[white_box] Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=True
    )
    
    # Training
    best_test_acc = 0.0
    best_test_epoch = 0
    patience = 40
    no_improve_count = 0
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Testing phase
        model.eval()
        test_correct = 0
        test_total = 0
        all_test_preds = []
        all_test_labels = []
        all_test_probs = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(device)
                labels = labels.to(device)
                
                outputs = model(features)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_test_preds.extend(predicted.cpu().numpy())
                all_test_labels.extend(labels.cpu().numpy())
                probs = torch.softmax(outputs, dim=1)
                all_test_probs.extend(probs[:, 1].cpu().numpy())
        
        test_acc = test_correct / test_total
        test_accs.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_acc)
        
        # Early stopping mechanism
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch + 1
            no_improve_count = 0
        else:
            no_improve_count += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
                  f"LR: {current_lr:.6f}, Best: {best_test_acc:.4f} @ Epoch {best_test_epoch}")
        
        # Early stopping check
        if no_improve_count >= patience:
            print(f"\n[white_box] Early stopping triggered")
            print(f"[white_box] Best test accuracy: {best_test_acc:.4f} @ Epoch {best_test_epoch}")
            break
    
    print(f"\n[white_box] Training completed! Best test accuracy: {best_test_acc:.4f} @ Epoch {best_test_epoch}")
    
    # Final evaluation
    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)
    all_test_probs = np.array(all_test_probs)
    
    final_test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_auc = roc_auc_score(all_test_labels, all_test_probs)
    
    precision = precision_score(all_test_labels, all_test_preds)
    recall = recall_score(all_test_labels, all_test_preds)
    f1 = f1_score(all_test_labels, all_test_preds)
    cm = confusion_matrix(all_test_labels, all_test_preds)
    
    member_mask = all_test_labels == 1
    non_member_mask = all_test_labels == 0
    
    member_acc = accuracy_score(all_test_labels[member_mask], all_test_preds[member_mask])
    non_member_acc = accuracy_score(all_test_labels[non_member_mask], all_test_preds[non_member_mask])
    
    print(f"\n[white_box] Final test accuracy: {final_test_acc:.4f}")
    print(f"[white_box] Member accuracy: {member_acc:.4f}")
    print(f"[white_box] Non-member accuracy: {non_member_acc:.4f}")
    print(f"[white_box] Precision: {precision:.4f}")
    print(f"[white_box] Recall: {recall:.4f}")
    print(f"[white_box] F1-score: {f1:.4f}")
    print(f"[white_box] ROC AUC: {test_auc:.4f}")
    print(f"[white_box] Confusion matrix:")
    print(f"  TN: {cm[0, 0]}, FP: {cm[0, 1]}")
    print(f"  FN: {cm[1, 0]}, TP: {cm[1, 1]}")
    
    return model, scaler, train_losses, train_accs, test_accs, all_test_labels, all_test_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints/ckpt_bert_A_finetune_max512")
    ap.add_argument("--model_type", choices=["baseline", "dann"], default="baseline")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--train_jsonl", default="corpus_A/A_train_leaky.jsonl")
    ap.add_argument("--test_jsonl", default="corpus_A/A_test_leaky.jsonl")
    ap.add_argument("--train_size", type=int, default=2000, help="Attack training set size")
    ap.add_argument("--num_steps", type=int, default=5, help="Number of retraining steps")
    ap.add_argument("--batch_size", type=int, default=8, help="Feature extraction batch size")
    ap.add_argument("--lr", type=float, default=0.001, help="Retraining learning rate")
    ap.add_argument("--epochs", type=int, default=300, help="Number of epochs for attack classifier training")
    ap.add_argument("--classifier_lr", type=float, default=0.001, help="Classifier learning rate")
    ap.add_argument("--output_prefix", default="white_box_attack_v2")
    args = ap.parse_args()
    
    device = torch.device(args.device)
    
    # Parse paths
    ckpt_dir = Path(args.ckpt_dir)
    train_jsonl = Path(args.train_jsonl)
    test_jsonl = Path(args.test_jsonl)
    
    if not ckpt_dir.is_absolute():
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        ckpt_dir = project_root / ckpt_dir
        train_jsonl = project_root / train_jsonl if not train_jsonl.is_absolute() else train_jsonl
        test_jsonl = project_root / test_jsonl if not test_jsonl.is_absolute() else test_jsonl
    
    print(f"[white_box] Loading target model: {ckpt_dir}")
    tokenizer, model, max_length, _ = build_model_and_tokenizer(
        ckpt_dir, args.model_type, device
    )
    
    # Load data
    print(f"[white_box] Loading data...")
    X_train, y_train = load_jsonl_xy(str(train_jsonl))
    X_test, y_test = load_jsonl_xy(str(test_jsonl))
    
    X_all = X_train + X_test
    y_all = list(y_train) + list(y_test)
    member_labels = [1] * len(X_train) + [0] * len(X_test)
    
    print(f"[white_box] Total samples: {len(X_all)}")
    print(f"[white_box] Member: {sum(member_labels)}, Non-member: {len(member_labels) - sum(member_labels)}")
    
    # Random shuffle
    indices = np.arange(len(X_all))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    X_all = [X_all[i] for i in indices]
    y_all = [y_all[i] for i in indices]
    member_labels = [member_labels[i] for i in indices]
    
    # Split train/test sets
    train_size = min(args.train_size, len(X_all) - 200)
    X_train_attack = X_all[:train_size]
    y_train_attack = y_all[:train_size]
    member_train = member_labels[:train_size]
    
    X_test_attack = X_all[train_size:]
    y_test_attack = y_all[train_size:]
    member_test = member_labels[train_size:]
    
    print(f"[white_box] Attack training set: {len(X_train_attack)}, Test set: {len(X_test_attack)}")
    
    # Extract features
    print(f"\n[white_box] Extracting white-box features from training set...")
    train_features = extract_white_box_features(
        copy.deepcopy(model), tokenizer, X_train_attack, y_train_attack, 
        max_length, device, args.model_type, 
        num_steps=args.num_steps, batch_size=args.batch_size, lr=args.lr
    )
    
    print(f"\n[white_box] Extracting white-box features from test set...")
    test_features = extract_white_box_features(
        copy.deepcopy(model), tokenizer, X_test_attack, y_test_attack,
        max_length, device, args.model_type,
        num_steps=args.num_steps, batch_size=args.batch_size, lr=args.lr
    )
    
    # Train attack classifier
    print(f"\n[white_box] Training attack classifier...")
    attack_model, scaler, train_losses, train_accs, test_accs, test_labels_final, test_probs = train_white_box_classifier(
        train_features, np.array(member_train),
        test_features, np.array(member_test),
        device, epochs=args.epochs, batch_size=64, lr=args.classifier_lr
    )
    
    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # Save training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_losses, label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(train_accs, label="Train Acc")
    axes[1].plot(test_accs, label="Test Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Test Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    train_curve_path = results_dir / f"{args.output_prefix}_training_curve.png"
    plt.savefig(train_curve_path, dpi=200)
    plt.close()
    print(f"[white_box] Training curve saved: {train_curve_path}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels_final, test_probs)
    auc = roc_auc_score(test_labels_final, test_probs)
    
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("White-box Attack ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = results_dir / f"{args.output_prefix}_roc.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"[white_box] ROC curve saved: {roc_path}")
    
    # Save result summary
    summary = {
        "train_size": int(train_size),
        "test_size": int(len(X_test_attack)),
        "num_steps": int(args.num_steps),
        "feature_dim": int(train_features.shape[1]),
        "final_test_accuracy": round(
            float(accuracy_score(test_labels_final, (test_probs > 0.5).astype(int))), 4
        ),
        "test_roc_auc": round(float(auc), 4),
        "best_test_accuracy": round(float(max(test_accs)), 4),
        "train_curve_path": str(train_curve_path),
        "roc_path": str(roc_path),
        "ckpt_dir": str(ckpt_dir),
        "model_type": args.model_type,
    }
    
    summary_path = results_dir / f"{args.output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[white_box] Result summary saved: {summary_path}")


if __name__ == "__main__":
    main()

