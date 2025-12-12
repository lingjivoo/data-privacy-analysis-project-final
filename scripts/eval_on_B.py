#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在 Hospital B 的 test.jsonl 上评估 BERT / BERT-DANN 模型（不再使用 TF-IDF）。

- model_type = "baseline":
    加载在 A 上用 train_bert_nli_on_A.py 训练的 EncHead(BERT encoder + 分类头)
    ckpt_dir 下需要:
        - encoder_meta.json
        - model_best.pt 或 model_final.pt

- model_type = "dann":
    加载用 train_domain_adapt_A2B.py 训练的 BertDANN(BERT encoder + 分类头 + domain head)
    ckpt_dir 下需要:
        - encoder_meta.json
        - bert_dann_model.pt

输出：
    - B-test Accuracy / Macro-F1 / per-class F1
    - classification report
    - 追加一行到 results/eval_B_results.csv
"""

import json
import argparse
import csv
from pathlib import Path
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

from transformers import AutoTokenizer, AutoModel

LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ========= 数据集 & collate ========= #

def load_jsonl_xy(path):
    """从 JSONL 加载 sentence1/2 + gold_label"""
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            s = (o.get("sentence1", "") + " [SEP] " + o.get("sentence2", "")).strip()
            X.append(s)
            y.append(LABEL2ID[o["gold_label"]])
    return X, np.array(y, dtype=np.int64)


class TextDatasetXY(Dataset):
    def __init__(self, texts, y):
        self.texts = texts
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.y[idx]


def make_collate(tokenizer, max_length):
    """将 (text, label) batch 转成 (input_ids, attention_mask, labels)"""

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.stack(labels)
        return enc["input_ids"], enc["attention_mask"], labels

    return collate


# ========= 与训练脚本对齐的结构 ========= #

def mean_pooling(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """与 train_bert_nli_on_A.py 一致的 mean pooling"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class MLPHead(nn.Module):
    """与 train_bert_nli_on_A.py 相同的 MLP 分类头"""

    def __init__(self, input_dim, num_classes,
                 hidden_dims=None, dropout: float = 0.3,
                 use_batchnorm: bool = True):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            if use_batchnorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class LogisticHead(nn.Module):
    """单层线性分类头"""

    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, x):
        return self.fc(x)


class EncHead(nn.Module):
    """
    与 train_bert_nli_on_A.py 的 EncHead 一致：
    - encoder: BERT
    - head: LogisticHead 或 MLPHead
    """

    def __init__(self, encoder: AutoModel, head: nn.Module, pooling: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            sent = mean_pooling(out.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            sent = out.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.head(sent)


class GRL(torch.autograd.Function):
    """Gradient Reversal Layer（供 BertDANN 使用）"""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class BertDANN(nn.Module):
    """
    与 train_domain_adapt_A2B.py 中的 BertDANN 对齐：
      - encoder: BERT encoder
      - head:    NLI 分类头
      - dom:     域分类头（评估时不用）
    """

    def __init__(self, encoder: AutoModel, head: nn.Module,
                 feat_dim: int, pooling: str = "mean",
                 dom_hidden: int = 128):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pooling = pooling
        self.dom = nn.Sequential(
            nn.Linear(feat_dim, dom_hidden),
            nn.ReLU(),
            nn.Linear(dom_hidden, 2),
        )

    def encode(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            sent = mean_pooling(out.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            sent = out.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return sent

    def forward(self, input_ids, attention_mask,
                lambd: float = 0.0, return_domain: bool = True):
        sent = self.encode(input_ids, attention_mask)
        y_logits = self.head(sent)
        if not return_domain:
            return y_logits
        fd = GRL.apply(sent, lambd)
        d_logits = self.dom(fd)
        return y_logits, d_logits


# ========= 构建模型（baseline / dann） ========= #

def build_model_and_tokenizer(ckpt_dir: Path,
                              model_type: str,
                              device: torch.device):
    """
    从 ckpt_dir 构建 BERT 模型和 tokenizer：
      - baseline: EncHead + model_best.pt/model_final.pt
      - dann:     BertDANN + bert_dann_model.pt
    """
    meta_path = ckpt_dir / "encoder_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"encoder_meta.json not found in {ckpt_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    encoder_name = meta["encoder_name"]
    pooling = meta.get("pooling", "mean")
    hidden_size = int(meta["hidden_size"])
    head_type = meta.get("head_type", "logistic")
    hidden_dims = meta.get("hidden_dims", None)
    dropout = meta.get("dropout", 0.3)
    use_batchnorm = meta.get("use_batchnorm", True)
    max_length = meta.get("max_length", 512)

    print(f"[ckpt={ckpt_dir}] encoder_name={encoder_name}")
    print(f"[ckpt={ckpt_dir}] head_type={head_type}, hidden_size={hidden_size}, pooling={pooling}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    encoder = AutoModel.from_pretrained(encoder_name, use_safetensors=True)

    # 构建分类头
    if head_type == "logistic":
        head = LogisticHead(hidden_size, 3)
    else:
        head = MLPHead(
            input_dim=hidden_size,
            num_classes=3,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        )

    if model_type == "baseline":
        model = EncHead(encoder, head, pooling=pooling)
        ckpt_best = ckpt_dir / "model_best.pt"
        ckpt_final = ckpt_dir / "model_final.pt"
        if ckpt_best.exists():
            state = torch.load(ckpt_best, map_location=device)
            print(f"[baseline] Loading weights from {ckpt_best}")
        elif ckpt_final.exists():
            state = torch.load(ckpt_final, map_location=device)
            print(f"[baseline] Loading weights from {ckpt_final}")
        else:
            raise FileNotFoundError(
                f"Neither model_best.pt nor model_final.pt found in {ckpt_dir}"
            )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[baseline] Missing keys: {missing}")
        if unexpected:
            print(f"[baseline] Unexpected keys: {unexpected}")

    else:  # model_type == "dann"
        model = BertDANN(
            encoder=encoder,
            head=head,
            feat_dim=hidden_size,
            pooling=pooling,
            dom_hidden=128,
        )
        ckpt_path = ckpt_dir / "bert_dann_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"bert_dann_model.pt not found in {ckpt_dir}")
        state = torch.load(ckpt_path, map_location=device)
        print(f"[dann] Loading weights from {ckpt_path}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[dann] Missing keys (usually none): {missing}")
        if unexpected:
            print(f"[dann] Unexpected keys: {unexpected}")

    model.to(device)
    return tokenizer, model, max_length, meta


# ========= 评估 ========= #

def eval_on_B(model, tokenizer, X_txt, y,
              max_length: int,
              device: torch.device,
              model_type: str,
              batch_size: int = 64):
    """在 B-test 上评估模型，返回 accuracy, macro-F1, per-class F1, preds, labels"""
    ds = TextDatasetXY(X_txt, y)
    collate = make_collate(tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.eval()
    all_preds, all_labels = [], []
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

            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)
    return accuracy, f1_macro, f1_per_class, all_preds, all_labels


# ========= main ========= #

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--B_test_jsonl", required=True)
    ap.add_argument(
        "--ckpt_dir",
        required=True,
        help="包含 encoder_meta.json 以及 model_best.pt/model_final.pt (baseline) 或 bert_dann_model.pt (dann)",
    )
    ap.add_argument(
        "--model_type",
        choices=["baseline", "dann"],
        default="baseline",
    )
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    ap.add_argument(
        "--result-name",
        type = str,
        help="包含 encoder_meta.json 以及 model_best.pt/model_final.pt (baseline) 或 bert_dann_model.pt (dann)",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_dir = Path(args.ckpt_dir)

    # 1) 构建模型 + tokenizer
    tokenizer, model, max_length, meta = build_model_and_tokenizer(
        ckpt_dir, args.model_type, device
    )

    # 2) 加载 B-test 数据
    X_txt, y = load_jsonl_xy(args.B_test_jsonl)

    # 3) 评估
    accuracy, f1_macro, f1_per_class, preds, labels = eval_on_B(
        model, tokenizer, X_txt, y, max_length, device, args.model_type
    )

    print(f"\n[B-test] {args.model_type} Results (ckpt={args.ckpt_dir}):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(
        f"  Per-class F1: "
        f"entailment={f1_per_class[0]:.4f}, "
        f"contradiction={f1_per_class[1]:.4f}, "
        f"neutral={f1_per_class[2]:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            labels, preds,
            target_names=["entailment", "contradiction", "neutral"]
        )
    )

    # 4) 读取 DP 参数（优先 config.json，其次 encoder_meta.json）
    dp_epsilon = "none"
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            dp_epsilon = config.get("dp_epsilon", "none")
    else:
        # baseline BERT 的 DP 信息可能写在 encoder_meta.json 里
        e = meta.get("dp_epsilon", 0.0)
        if isinstance(e, (int, float)) and e > 0:
            dp_epsilon = e
        else:
            dp_epsilon = "none"

    # 5) 写入 results/eval_B_results.csv
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = os.path.join("results",    args.result_name + "_results.csv")
    results_file = Path(results_file)
    result_row = {
        "ckpt_dir": str(args.ckpt_dir),
        "model_type": args.model_type,
        "test_file": args.B_test_jsonl,
        "dp_epsilon": dp_epsilon,
        "B_test_acc": accuracy,
        "B_test_f1_macro": f1_macro,
        "B_test_f1_entailment": f1_per_class[0],
        "B_test_f1_contradiction": f1_per_class[1],
        "B_test_f1_neutral": f1_per_class[2],
    }

    file_exists = results_file.exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    print(f"\nResults saved to {results_file}")
