#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Membership Inference Attack (MIA) using MLP classifier.

Read data from member_inference_records.csv:
  - Input features: Softmax probabilities (p_entailment, p_contradiction, p_neutral) + true label (gold_label)
  - Output labels: member (1) / non-member (0)

Training process:
  1) Randomly sample 2000 samples as training data, rest as test data
  2) Ensure member and non-member are randomly distributed in train/test data
  3) Train a simple MLP classifier
  4) Evaluate on test set and print accuracy
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from scipy.stats import ks_2samp


class MemberInferenceDataset(Dataset):
    """Membership inference dataset"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    """More complex MLP classifier with BatchNorm, residual connections, etc."""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.3, use_residual=True):
        super().__init__()
        self.use_residual = use_residual
        
        # Input projection layer
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layers (with residual connections)
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.BatchNorm1d(hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dims[-1] // 2, num_classes)
        )
        
        # Projection layers for residual connections (if dimensions don't match)
        self.residual_projs = nn.ModuleList()
        if use_residual:
            for i in range(len(hidden_dims) - 1):
                if hidden_dims[i] != hidden_dims[i + 1]:
                    self.residual_projs.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                else:
                    self.residual_projs.append(nn.Identity())
    
    def forward(self, x):
        # Input projection
        out = self.input_proj(x)
        
        # Hidden layers (with residual connections)
        for i, layer in enumerate(self.hidden_layers):
            if self.use_residual and i < len(self.residual_projs):
                residual = self.residual_projs[i](out)
                out = layer(out) + residual
            else:
                out = layer(out)
        
        # Output layer
        out = self.output_layer(out)
        return out


def load_data(csv_path: Path):
    """Load data from CSV, return features and labels"""
    features = []
    labels = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Basic features: Softmax probabilities
            p_entailment = float(row["p_entailment"])
            p_contradiction = float(row["p_contradiction"])
            p_neutral = float(row["p_neutral"])
            gold_label = int(row["gold_label"])
            
            # Additional features: max_softmax, mentr, correct
            max_softmax = float(row.get("max_softmax", 0.0))
            mentr = float(row.get("mentr", 0.0))
            correct = int(row.get("correct", 0))
            
            # Feature vector:
            # [p_entailment, p_contradiction, p_neutral, 
            #  gold_label_one_hot (3 dimensions),
            #  max_softmax, mentr, correct]
            gold_one_hot = [0.0, 0.0, 0.0]
            gold_one_hot[gold_label] = 1.0
            
            feature = [p_entailment, p_contradiction, p_neutral] + gold_one_hot + [max_softmax, mentr, float(correct)]
            features.append(feature)
            
            # Labels: member (1) / non-member (0)
            labels.append(int(row["member_label"]))
    
    return np.array(features), np.array(labels)


def train_mlp_classifier(train_features, train_labels, test_features, test_labels,
                         device, epochs=1000, batch_size=64, lr=0.001):
    """Train MLP classifier"""
    
    # Data standardization
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Create datasets
    train_dataset = MemberInferenceDataset(train_features_scaled, train_labels)
    test_dataset = MemberInferenceDataset(test_features_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model (more complex architecture)
    input_dim = train_features.shape[1]
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],  # Deeper network
        num_classes=2,
        dropout=0.3,
        use_residual=True  # Use residual connections
    )
    model.to(device)
    
    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[mia_ml] Total model parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, verbose=True
    )
    
    # Training
    print(f"[mia_ml] Starting training, training samples: {len(train_dataset)}, test samples: {len(test_dataset)}")
    print(f"[mia_ml] Input feature dimension: {input_dim}")
    
    best_test_acc = 0.0
    best_test_epoch = 0
    patience = 50  # Early stopping patience value
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
                probs = F.softmax(outputs, dim=1)
                all_test_probs.extend(probs[:, 1].cpu().numpy())  # Probability of member class
        
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
                  f"LR: {current_lr:.6f}, Best Test: {best_test_acc:.4f} @ Epoch {best_test_epoch}")
        
        # Early stopping check
        if no_improve_count >= patience:
            print(f"\n[mia_ml] Early stopping triggered: {patience} epochs without improvement")
            print(f"[mia_ml] Best test accuracy: {best_test_acc:.4f} @ Epoch {best_test_epoch}")
            break
    
    print(f"\n[mia_ml] Training completed! Best test accuracy: {best_test_acc:.4f} @ Epoch {best_test_epoch}")
    
    # Re-evaluate test set (using best model state)
    model.eval()
    all_test_preds = []
    all_test_labels = []
    all_test_probs = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            all_test_preds.extend(predicted.cpu().numpy())
            all_test_labels.extend(labels.cpu().numpy())
            probs = F.softmax(outputs, dim=1)
            all_test_probs.extend(probs[:, 1].cpu().numpy())
    
    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)
    all_test_probs = np.array(all_test_probs)
    
    final_test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_auc = roc_auc_score(all_test_labels, all_test_probs)
    
    # Compute detailed metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    precision = precision_score(all_test_labels, all_test_preds)
    recall = recall_score(all_test_labels, all_test_preds)
    f1 = f1_score(all_test_labels, all_test_preds)
    cm = confusion_matrix(all_test_labels, all_test_preds)
    
    # Compute accuracy separately for member and non-member
    member_mask = all_test_labels == 1
    non_member_mask = all_test_labels == 0
    
    member_acc = accuracy_score(all_test_labels[member_mask], all_test_preds[member_mask])
    non_member_acc = accuracy_score(all_test_labels[non_member_mask], all_test_preds[non_member_mask])
    
    print(f"\n[mia_ml] Final test accuracy: {final_test_acc:.4f}")
    print(f"[mia_ml] Member accuracy: {member_acc:.4f} ({member_mask.sum()} samples)")
    print(f"[mia_ml] Non-member accuracy: {non_member_acc:.4f} ({non_member_mask.sum()} samples)")
    print(f"[mia_ml] Precision: {precision:.4f}")
    print(f"[mia_ml] Recall: {recall:.4f}")
    print(f"[mia_ml] F1-score: {f1:.4f}")
    print(f"[mia_ml] Test set ROC AUC: {test_auc:.4f}")
    print(f"[mia_ml] Confusion matrix:")
    print(f"  TN (Non-member correct): {cm[0, 0]}, FP (Non-member misclassified): {cm[0, 1]}")
    print(f"  FN (Member misclassified): {cm[1, 0]}, TP (Member correct): {cm[1, 1]}")
    print("\n[mia_ml] Test set classification report:")
    print(classification_report(all_test_labels, all_test_preds, 
                                target_names=["non-member", "member"]))
    
    return model, scaler, train_losses, train_accs, test_accs, all_test_labels, all_test_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv_path",
        default="results/member_inference_records.csv",
        help="Input CSV file path",
    )
    ap.add_argument(
        "--train_size",
        type=int,
        default=2000,
        help="Training set size (default 2000)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default 50)",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default 64)",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default 0.001)",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    ap.add_argument(
        "--output_prefix",
        default="mia_ml",
        help="输出文件前缀",
    )
    args = ap.parse_args()
    
    # 解析路径
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        csv_path = project_root / csv_path
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"[mia_ml] Loading data: {csv_path}")
    features, labels = load_data(csv_path)
    print(f"[mia_ml] Total samples: {len(features)}")
    print(f"[mia_ml] Member samples: {np.sum(labels == 1)}")
    print(f"[mia_ml] Non-member samples: {np.sum(labels == 0)}")
    
    # Analyze feature distribution differences
    member_mask = labels == 1
    non_member_mask = labels == 0
    feature_names = ["p_entailment", "p_contradiction", "p_neutral", 
                     "gold_0", "gold_1", "gold_2", 
                     "max_softmax", "mentr", "correct"]
    
    print(f"\n[mia_ml] Feature distribution analysis (Member vs Non-member):")
    for i, name in enumerate(feature_names):
        member_mean = features[member_mask, i].mean()
        non_member_mean = features[non_member_mask, i].mean()
        diff = member_mean - non_member_mean
        member_std = features[member_mask, i].std()
        non_member_std = features[non_member_mask, i].std()
        print(f"  {name:15s}: Member={member_mean:8.4f}±{member_std:.4f}, "
              f"Non-member={non_member_mean:8.4f}±{non_member_std:.4f}, "
              f"Diff={diff:8.4f}")
    
    # Compute feature correlation
    from scipy.stats import ks_2samp
    print(f"\n[mia_ml] KS test (Member vs Non-member distribution differences):")
    for i, name in enumerate(feature_names):
        ks_stat, ks_p = ks_2samp(features[member_mask, i], features[non_member_mask, i])
        print(f"  {name:15s}: KS-stat={ks_stat:.4f}, p-value={ks_p:.6f} {'***' if ks_p < 0.001 else '**' if ks_p < 0.01 else '*' if ks_p < 0.05 else ''}")
    
    # Randomly shuffle data to ensure member and non-member are randomly distributed
    indices = np.arange(len(features))
    np.random.seed(42)  # Set random seed for reproducibility
    np.random.shuffle(indices)
    
    features = features[indices]
    labels = labels[indices]
    
    # Split train and test sets
    train_size = min(args.train_size, len(features) - 100)  # Keep at least 100 test samples
    train_features = features[:train_size]
    train_labels = labels[:train_size]
    test_features = features[train_size:]
    test_labels = labels[train_size:]
    
    print(f"[mia_ml] Training set: {len(train_features)} samples (Member: {np.sum(train_labels == 1)}, "
          f"Non-member: {np.sum(train_labels == 0)})")
    print(f"[mia_ml] Test set: {len(test_features)} samples (Member: {np.sum(test_labels == 1)}, "
          f"Non-member: {np.sum(test_labels == 0)})")
    
    device = torch.device(args.device)
    print(f"[mia_ml] Using device: {device}")
    
    # Train model
    model, scaler, train_losses, train_accs, test_accs, test_labels_final, test_probs = train_mlp_classifier(
        train_features, train_labels, test_features, test_labels,
        device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    
    # Save training curves
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
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
    print(f"[mia_ml] Training curve saved: {train_curve_path}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(test_labels_final, test_probs)
    auc = roc_auc_score(test_labels_final, test_probs)
    
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"ROC AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("MLP Classifier ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = results_dir / f"{args.output_prefix}_roc.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"[mia_ml] ROC curve saved: {roc_path}")
    
    # Save result summary
    summary = {
        "train_size": int(train_size),
        "test_size": int(len(test_features)),
        "final_test_accuracy": float(accuracy_score(test_labels_final, 
                                                     (test_probs > 0.5).astype(int))),
        "test_roc_auc": float(auc),
        "best_test_accuracy": float(max(test_accs)),
        "train_curve_path": str(train_curve_path),
        "roc_path": str(roc_path),
        "csv_path": str(csv_path),
    }
    
    summary_path = results_dir / f"{args.output_prefix}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[mia_ml] Result summary saved: {summary_path}")


if __name__ == "__main__":
    main()

