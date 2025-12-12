#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
白盒成员推断攻击（White-box MIA）。

通过再训练模型，记录训练过程中的 loss、梯度等信息作为特征，
训练一个分类器来区分 member 和 non-member。

流程：
  1) 从 member_inference_records.csv 读取数据，划分训练/测试集（与 mia_ml 相同）
  2) 对每个样本进行前向传播和反向传播（再训练几步）
  3) 记录 loss、梯度范数、参数更新等信息作为特征
  4) 使用这些特征训练一个分类器来预测 member/non-member
"""

import argparse
import csv
import json
from pathlib import Path

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

# 同目录下的推理工具
from eval_on_B import (
    LABEL2ID,
    TextDatasetXY,
    build_model_and_tokenizer,
    load_jsonl_xy,
    make_collate,
)


class WhiteBoxFeatureDataset(Dataset):
    """白盒特征数据集"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class WhiteBoxClassifier(nn.Module):
    """用于白盒攻击的分类器"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64], num_classes=2, dropout=0.3):
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


def extract_white_box_features(model, tokenizer, X_txt, y, max_length, device, 
                               model_type, num_steps=3, batch_size=1):
    """
    对每个样本进行再训练，提取白盒特征（loss、梯度等）。
    
    Args:
        model: 目标模型
        tokenizer: tokenizer
        X_txt: 文本列表
        y: 标签列表
        max_length: 最大长度
        device: 设备
        model_type: 模型类型（baseline/dann）
        num_steps: 每个样本再训练的步数
        batch_size: 批次大小（通常为1，因为每个样本单独训练）
    
    Returns:
        features: numpy array of shape [num_samples, feature_dim]
    """
    model.train()  # 设置为训练模式
    
    # 创建数据集和dataloader
    ds = TextDatasetXY(X_txt, y)
    collate = make_collate(tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    
    criterion = nn.CrossEntropyLoss()
    all_features = []
    
    print(f"[white_box] 开始提取白盒特征，样本数: {len(X_txt)}, 每个样本训练 {num_steps} 步...")
    print(f"[white_box] 预计总时间较长，请耐心等待...")
    
    for sample_idx, (input_ids, attn_mask, yb) in enumerate(dl):
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        yb = yb.to(device)
        
        # 保存初始参数（用于计算参数更新）
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # 记录特征
        sample_features = []
        losses = []
        grad_norms = []
        grad_nz_ratios = []
        logits_feats = []  # 每步的分布特征（entropy/max_prob/top2_gap/prob_std）
        
        # 对当前样本进行 num_steps 步训练
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # 前向传播
            if model_type == "baseline":
                logits = model(input_ids, attn_mask)
            else:  # dann
                logits = model(input_ids, attn_mask, lambd=0.0, return_domain=False)
            
            probs = torch.softmax(logits, dim=1)
            probs_np = probs.detach().cpu().numpy()
            # 仅 batch_size=1 时，直接取第一个样本
            p = probs_np[0]
            sorted_p = np.sort(p)[::-1]
            top1 = sorted_p[0]
            top2 = sorted_p[1] if len(sorted_p) > 1 else 0.0
            entropy = -np.sum(p * (np.log(p + 1e-12)))
            logits_feats.append([
                entropy,
                top1,
                top1 - top2,
                p.std(),
            ])

            loss = criterion(logits, yb)
            
            # 反向传播
            loss.backward()
            
            # 记录 loss
            losses.append(loss.item())
            
            # 计算梯度范数
            total_grad_norm = 0.0
            total_elements = 0
            nonzero_grad = 0
            param_count = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += param_grad_norm ** 2
                    param_count += 1
                    g = param.grad.data
                    total_elements += g.numel()
                    nonzero_grad += (g != 0).sum().item()
            grad_norm = (total_grad_norm ** 0.5) if param_count > 0 else 0.0
            grad_norms.append(grad_norm)
            nz_ratio = (nonzero_grad / total_elements) if total_elements > 0 else 0.0
            grad_nz_ratios.append(nz_ratio)
            
            # 更新参数
            optimizer.step()
        
        # 计算参数更新量（最终参数 - 初始参数）
        param_updates = []
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_params:
                update = (param.data - initial_params[name]).norm(2).item()
                param_updates.append(update)
        
        # 恢复初始参数（避免影响下一个样本）
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_params:
                param.data.copy_(initial_params[name])
        
        # 构建特征向量
        # 特征包括：
        # 1. 每步的 loss (num_steps 维)
        # 2. 每步的梯度范数 (num_steps 维)
        # 3. loss 的统计量：mean, std, min, max, final (5维)
        # 4. 梯度范数的统计量：mean, std, min, max, final (5维)
        # 5. 参数更新量的统计量：mean, std, min, max (4维)
        feature_vec = []
        
        # 每步的 loss
        feature_vec.extend(losses)
        
        # 每步的梯度范数
        feature_vec.extend(grad_norms)
        
        # loss 统计量
        if len(losses) > 0:
            feature_vec.extend([
                np.mean(losses),
                np.std(losses) if len(losses) > 1 else 0.0,
                np.min(losses),
                np.max(losses),
                losses[-1]  # final loss
            ])
        else:
            feature_vec.extend([0.0] * 5)
        
        # 梯度范数统计量
        if len(grad_norms) > 0:
            feature_vec.extend([
                np.mean(grad_norms),
                np.std(grad_norms) if len(grad_norms) > 1 else 0.0,
                np.min(grad_norms),
                np.max(grad_norms),
                grad_norms[-1]  # final grad norm
            ])
        else:
            feature_vec.extend([0.0] * 5)
        
        # 参数更新量统计量
        if len(param_updates) > 0:
            param_updates_arr = np.array(param_updates)
            feature_vec.extend([
                param_updates_arr.mean(),
                param_updates_arr.std() if len(param_updates_arr) > 1 else 0.0,
                param_updates_arr.min(),
                param_updates_arr.max(),
                np.median(param_updates_arr),
                np.percentile(param_updates_arr, 90),
            ])
        else:
            feature_vec.extend([0.0] * 6)

        # 每步的梯度非零比例
        feature_vec.extend(grad_nz_ratios if len(grad_nz_ratios) > 0 else [0.0] * num_steps)
        # 梯度非零比例统计量
        if len(grad_nz_ratios) > 0:
            feature_vec.extend([
                np.mean(grad_nz_ratios),
                np.std(grad_nz_ratios) if len(grad_nz_ratios) > 1 else 0.0,
                np.min(grad_nz_ratios),
                np.max(grad_nz_ratios),
                grad_nz_ratios[-1],
            ])
        else:
            feature_vec.extend([0.0] * 5)

        # 每步的输出分布特征（entropy, max_prob, top2_gap, prob_std）
        if len(logits_feats) > 0:
            logits_arr = np.array(logits_feats)  # shape [num_steps, 4]
            feature_vec.extend(logits_arr.flatten().tolist())
            # 分布特征统计量（逐列）
            feature_vec.extend(np.mean(logits_arr, axis=0).tolist())
            feature_vec.extend(np.std(logits_arr, axis=0).tolist())
            feature_vec.extend(np.min(logits_arr, axis=0).tolist())
            feature_vec.extend(np.max(logits_arr, axis=0).tolist())
            feature_vec.extend(logits_arr[-1].tolist())  # 最后一步
        else:
            feature_vec.extend([0.0] * (num_steps * 4))  # 每步4个
            feature_vec.extend([0.0] * 4)  # mean
            feature_vec.extend([0.0] * 4)  # std
            feature_vec.extend([0.0] * 4)  # min
            feature_vec.extend([0.0] * 4)  # max
            feature_vec.extend([0.0] * 4)  # final
        
        all_features.append(feature_vec)
        
        # 每10个样本输出一次进度
        if (sample_idx + 1) % 10 == 0:
            progress = (sample_idx + 1) / len(dl) * 100
            print(f"[white_box] 特征提取进度: {sample_idx + 1}/{len(dl)} ({progress:.1f}%)")
        # 每100个样本输出一次详细进度
        elif (sample_idx + 1) % 100 == 0:
            progress = (sample_idx + 1) / len(dl) * 100
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
            avg_grad = np.mean(grad_norms) if len(grad_norms) > 0 else 0.0
            print(f"[white_box] 已处理 {sample_idx + 1}/{len(dl)} 个样本 ({progress:.1f}%) - "
                  f"平均Loss: {avg_loss:.4f}, 平均梯度范数: {avg_grad:.4f}")
    
    print(f"[white_box] 特征提取完成！")
    print(f"[white_box] 总样本数: {len(all_features)}, 特征维度: {len(all_features[0])}")
    
    # 输出特征统计信息
    features_array = np.array(all_features)
    print(f"[white_box] 特征统计:")
    print(f"  Loss特征范围: [{features_array[:, :num_steps].min():.4f}, {features_array[:, :num_steps].max():.4f}]")
    print(f"  梯度范数范围: [{features_array[:, num_steps:2*num_steps].min():.4f}, {features_array[:, num_steps:2*num_steps].max():.4f}]")
    # 梯度非零比例位置：前面已有 2*num_steps + 14（loss/gradnorm stats + param updates 6）
    grad_nz_start = 2 * num_steps + 14
    grad_nz_end = grad_nz_start + num_steps
    print(f"  梯度非零比例范围: [{features_array[:, grad_nz_start:grad_nz_end].min():.4f}, "
          f"{features_array[:, grad_nz_start:grad_nz_end].max():.4f}]")
    
    return features_array


def load_data_from_csv(csv_path: Path):
    """从 CSV 加载数据，返回文本、标签和成员标签"""
    texts = []
    labels = []
    member_labels = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 注意：CSV 中没有原始文本，我们需要从原始数据文件加载
            # 这里先记录索引，稍后从原始文件加载
            member_labels.append(int(row["member_label"]))
            labels.append(int(row["gold_label"]))
    
    return member_labels, labels


def load_original_data(train_jsonl: Path, test_jsonl: Path):
    """从原始 JSONL 文件加载文本和标签"""
    X_train, y_train = load_jsonl_xy(str(train_jsonl))
    X_test, y_test = load_jsonl_xy(str(test_jsonl))
    
    X_all = X_train + X_test
    y_all = list(y_train) + list(y_test)
    
    return X_all, y_all


def train_white_box_classifier(train_features, train_labels, test_features, test_labels,
                               device, epochs=500, batch_size=64, lr=0.001):
    """训练白盒攻击分类器"""
    
    # 数据标准化
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # 创建数据集
    train_dataset = WhiteBoxFeatureDataset(train_features_scaled, train_labels)
    test_dataset = WhiteBoxFeatureDataset(test_features_scaled, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    input_dim = train_features.shape[1]
    model = WhiteBoxClassifier(
        input_dim=input_dim,
        hidden_dims=[128, 64],
        num_classes=2,
        dropout=0.3
    )
    model.to(device)
    
    print(f"[white_box] 分类器输入特征维度: {input_dim}")
    print(f"[white_box] 训练样本数: {len(train_dataset)}, 测试样本数: {len(test_dataset)}")
    
    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=20, verbose=True
    )
    
    # 训练
    best_test_acc = 0.0
    best_test_epoch = 0
    patience = 50
    no_improve_count = 0
    train_losses = []
    train_accs = []
    test_accs = []
    
    for epoch in range(epochs):
        # 训练阶段
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
        
        # 测试阶段
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
        
        # 学习率调度
        scheduler.step(test_acc)
        
        # 早停机制
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
        
        # 早停检查
        if no_improve_count >= patience:
            print(f"\n[white_box] 早停触发：{patience} 个 epoch 没有改进")
            print(f"[white_box] 最佳测试准确率: {best_test_acc:.4f} @ Epoch {best_test_epoch}")
            break
    
    print(f"\n[white_box] 训练完成！最佳测试准确率: {best_test_acc:.4f} @ Epoch {best_test_epoch}")
    
    # 最终评估
    all_test_preds = np.array(all_test_preds)
    all_test_labels = np.array(all_test_labels)
    all_test_probs = np.array(all_test_probs)
    
    final_test_acc = accuracy_score(all_test_labels, all_test_preds)
    test_auc = roc_auc_score(all_test_labels, all_test_probs)
    
    # 计算详细指标
    precision = precision_score(all_test_labels, all_test_preds)
    recall = recall_score(all_test_labels, all_test_preds)
    f1 = f1_score(all_test_labels, all_test_preds)
    cm = confusion_matrix(all_test_labels, all_test_preds)
    
    # 分别计算 member 和 non-member 的准确率
    member_mask = all_test_labels == 1
    non_member_mask = all_test_labels == 0
    
    member_acc = accuracy_score(all_test_labels[member_mask], all_test_preds[member_mask])
    non_member_acc = accuracy_score(all_test_labels[non_member_mask], all_test_preds[non_member_mask])
    
    print(f"\n[white_box] 最终测试准确率: {final_test_acc:.4f}")
    print(f"[white_box] Member 准确率: {member_acc:.4f} ({member_mask.sum()} 个样本)")
    print(f"[white_box] Non-member 准确率: {non_member_acc:.4f} ({non_member_mask.sum()} 个样本)")
    print(f"[white_box] Precision: {precision:.4f}")
    print(f"[white_box] Recall: {recall:.4f}")
    print(f"[white_box] F1-score: {f1:.4f}")
    print(f"[white_box] 测试集 ROC AUC: {test_auc:.4f}")
    print(f"[white_box] 混淆矩阵:")
    print(f"  TN (Non-member正确): {cm[0, 0]}, FP (Non-member误判): {cm[0, 1]}")
    print(f"  FN (Member误判): {cm[1, 0]}, TP (Member正确): {cm[1, 1]}")
    print("\n[white_box] 测试集分类报告:")
    print(classification_report(
        all_test_labels,
        all_test_preds,
        target_names=["non-member", "member"],
        digits=4,
    ))
    
    return model, scaler, train_losses, train_accs, test_accs, all_test_labels, all_test_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt_dir",
        default="checkpoints/ckpt_bert_A_finetune_max512",
        help="包含 encoder_meta.json 与权重的目录",
    )
    ap.add_argument("--model_type", choices=["baseline", "dann"], default="baseline")
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument(
        "--csv_path",
        default="results/member_inference_records.csv",
        help="输入的 CSV 文件路径",
    )
    ap.add_argument(
        "--train_jsonl",
        default="corpus_A/A_train_leaky.jsonl",
        help="训练数据（成员样本）",
    )
    ap.add_argument(
        "--test_jsonl",
        default="corpus_A/A_test_leaky.jsonl",
        help="测试数据（非成员样本）",
    )
    ap.add_argument(
        "--train_size",
        type=int,
        default=2000,
        help="攻击分类器训练集大小（默认2000）",
    )
    ap.add_argument(
        "--num_steps",
        type=int,
        default=3,
        help="每个样本再训练的步数（默认3）",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="攻击分类器训练轮数（默认500）",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小（默认64）",
    )
    ap.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="学习率（默认0.001）",
    )
    ap.add_argument(
        "--output_prefix",
        default="white_box_attack",
        help="输出文件前缀",
    )
    args = ap.parse_args()
    
    device = torch.device(args.device)
    
    # 解析路径
    ckpt_dir = Path(args.ckpt_dir)
    csv_path = Path(args.csv_path)
    train_jsonl = Path(args.train_jsonl)
    test_jsonl = Path(args.test_jsonl)
    
    if not ckpt_dir.is_absolute():
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        ckpt_dir = project_root / ckpt_dir
        csv_path = project_root / csv_path if not csv_path.is_absolute() else csv_path
        train_jsonl = project_root / train_jsonl if not train_jsonl.is_absolute() else train_jsonl
        test_jsonl = project_root / test_jsonl if not test_jsonl.is_absolute() else test_jsonl
    
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"[white_box] 加载目标模型: {ckpt_dir}")
    tokenizer, model, max_length, _ = build_model_and_tokenizer(
        ckpt_dir, args.model_type, device
    )
    
    # 加载原始数据
    print(f"[white_box] 加载原始数据...")
    X_train, y_train = load_jsonl_xy(str(train_jsonl))
    X_test, y_test = load_jsonl_xy(str(test_jsonl))
    
    # 合并数据
    X_all = X_train + X_test
    y_all = list(y_train) + list(y_test)
    member_labels = [1] * len(X_train) + [0] * len(X_test)
    
    print(f"[white_box] 总样本数: {len(X_all)}")
    print(f"[white_box] Member 样本数: {sum(member_labels)}")
    print(f"[white_box] Non-member 样本数: {len(member_labels) - sum(member_labels)}")
    
    # 随机打乱数据（与 mia_ml 相同）
    indices = np.arange(len(X_all))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    X_all = [X_all[i] for i in indices]
    y_all = [y_all[i] for i in indices]
    member_labels = [member_labels[i] for i in indices]
    
    # 划分训练集和测试集（与 mia_ml 相同）
    train_size = min(args.train_size, len(X_all) - 100)
    X_train_attack = X_all[:train_size]
    y_train_attack = y_all[:train_size]
    member_train = member_labels[:train_size]
    
    X_test_attack = X_all[train_size:]
    y_test_attack = y_all[train_size:]
    member_test = member_labels[train_size:]
    
    print(f"[white_box] 攻击分类器训练集: {len(X_train_attack)} 条")
    print(f"[white_box] 攻击分类器测试集: {len(X_test_attack)} 条")
    
    # 提取白盒特征
    print(f"\n[white_box] 提取训练集白盒特征...")
    train_features = extract_white_box_features(
        model, tokenizer, X_train_attack, y_train_attack, max_length,
        device, args.model_type, num_steps=args.num_steps, batch_size=1
    )
    
    print(f"\n[white_box] 提取测试集白盒特征...")
    test_features = extract_white_box_features(
        model, tokenizer, X_test_attack, y_test_attack, max_length,
        device, args.model_type, num_steps=args.num_steps, batch_size=1
    )
    
    # 训练攻击分类器
    print(f"\n[white_box] 训练攻击分类器...")
    attack_model, scaler, train_losses, train_accs, test_accs, test_labels_final, test_probs = train_white_box_classifier(
        train_features, np.array(member_train),
        test_features, np.array(member_test),
        device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )
    
    # 保存结果
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # 保存训练曲线
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
    print(f"[white_box] 训练曲线已保存: {train_curve_path}")
    
    # 画 ROC 曲线
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
    print(f"[white_box] ROC 曲线已保存: {roc_path}")
    
    # 保存结果摘要
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
    print(f"[white_box] 结果摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()

