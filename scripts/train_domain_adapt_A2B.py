#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain Adaptation: BERT-based DANN / DP-DANN

- Backbone: 与 train_bert_nli_on_A.py 相同的 BERT encoder + (logistic/MLP) head
- 初始化：从在 Hospital A 上预训练过的 checkpoint 中加载 (encoder + head) 权重
- Domain head: 在 sentence embedding 上接一个 MLP + Gradient Reversal Layer (GRL)
- 训练数据：
    * 源域 A: 有 NLI 标签 (entailment / contradiction / neutral)
    * 目标域 B: 无标签（训练时），只参与 domain 分类
- 验证 / Early stop:
    * A_val: 用于监控源域性能（可选分析）
    * B_val: 有标签，但仅用于验证与 early stopping，不参与训练
- 可选：Opacus DP-SGD

输入/输出：
  * 输入 JSONL:
      - --A_train_jsonl:  A_train 带 gold_label
      - --A_val_jsonl:    A_val  带 gold_label（仅用于监控 A 上的分类精度）
      - --B_train_jsonl:  B_train 无标签，只参与 domain 分类
      - --B_val_jsonl:    B_val  带 gold_label，用于早停与选择 B 最优模型
  * 从 --pretrained_ckpt_dir 读取：
      - encoder_meta.json
      - model_best.pt (优先) 或 model_final.pt
  * 保存到 --out_dir：
      - bert_dann_model.pt      （在 B_val 上 acc 最好的 checkpoint）
      - encoder_meta.json
      - config.json
      - train_history.csv
      - 并向 results/dann_A2B_results.csv 追加一行结果
"""

import json
import argparse
import math
import random
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModel

# ------------------ 常量与随机种子 ------------------ #

SEED = 2025
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ------------------ 数据加载 ------------------ #

def load_jsonl_xy(path):
    """加载 A/B 域：sentence1, sentence2, gold_label -> (texts, labels)"""
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            s = (o.get("sentence1", "") + " [SEP] " +
                 o.get("sentence2", "")).strip()
            X.append(s)
            y.append(LABEL2ID[o["gold_label"]])
    return X, np.array(y, dtype=np.int64)


def load_jsonl_x(path):
    """加载 B_train：只取 sentence1 + [SEP] + sentence2"""
    X = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            s = (o.get("sentence1", "") + " [SEP] " +
                 o.get("sentence2", "")).strip()
            X.append(s)
    return X


class TextDatasetXY(Dataset):
    """带标签文本 Dataset，用于 A_train / A_val / B_val"""

    def __init__(self, texts, y):
        self.texts = texts
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.y[idx]


def make_collate_src(tokenizer, max_length):
    """batch: (input_ids, attention_mask, labels)"""

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding="max_length",      # ✅ 固定 pad 到 max_length
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.stack(labels)
        return enc["input_ids"], enc["attention_mask"], labels

    return collate


# ------------------ BERT 结构（与 baseline 对齐） ------------------ #

def mean_pooling(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """与 train_bert_nli_on_A.py 同样的 mean pooling 实现"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class MLPHead(nn.Module):
    """与 train_bert_nli_on_A.py 相同的多层 MLP 头"""

    def __init__(self, input_dim, num_classes, hidden_dims=None,
                 dropout: float = 0.3, use_batchnorm: bool = True):
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


# ------------------ GRL + BERT-DANN 模型 ------------------ #

class GRL(torch.autograd.Function):
    """Gradient Reversal Layer"""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class BertDANN(nn.Module):
    """
    BERT-based DANN:
      - encoder: 预训练好的 BERT encoder
      - head:    NLI 分类头（logistic / MLP）
      - dom:     domain classifier (2-way: source=0, target=1)
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
        """
        return_domain=True: 返回 (task_logits, domain_logits)
        return_domain=False: 只返回 task_logits（评估用）
        """
        sent = self.encode(input_ids, attention_mask)
        y_logits = self.head(sent)
        if not return_domain:
            return y_logits

        fd = GRL.apply(sent, lambd)
        d_logits = self.dom(fd)
        return y_logits, d_logits


# ------------------ 从 A 上预训练的 checkpoint 构建 BERT-DANN ------------------ #

def build_bert_dann_from_pretrained(pretrained_ckpt_dir: Path,
                                    device: torch.device):
    """
    从 train_bert_nli_on_A.py 训练得到的 ckpt_dir 中加载：
      - encoder_meta.json
      - model_best.pt (优先) 或 model_final.pt
    并构建 BertDANN 模型（encoder+head 权重继承，dom head 随机初始化）。
    """
    meta_path = pretrained_ckpt_dir / "encoder_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"encoder_meta.json not found in {pretrained_ckpt_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    encoder_name = meta["encoder_name"]
    pooling = meta.get("pooling", "mean")
    hidden_size = int(meta["hidden_size"])
    head_type = meta.get("head_type", "logistic")
    hidden_dims = meta.get("hidden_dims", None)
    dropout = meta.get("dropout", 0.3)
    use_batchnorm = meta.get("use_batchnorm", True)

    print(f"[Pretrained] encoder_name={encoder_name}")
    print(f"[Pretrained] head_type={head_type}, hidden_size={hidden_size}, pooling={pooling}")

    tokenizer = AutoTokenizer.from_pretrained(encoder_name, use_fast=True)
    encoder = AutoModel.from_pretrained(encoder_name, use_safetensors=True)

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

    model = BertDANN(
        encoder=encoder,
        head=head,
        feat_dim=hidden_size,
        pooling=pooling,
        dom_hidden=128,
    )

    # 加载 A 域预训练权重：只覆盖 encoder + head，dom head 随机初始化
    ckpt_best = pretrained_ckpt_dir / "model_best.pt"
    ckpt_final = pretrained_ckpt_dir / "model_final.pt"
    if ckpt_best.exists():
        state = torch.load(ckpt_best, map_location=device)
        print(f"[Pretrained] Loading weights from {ckpt_best}")
    elif ckpt_final.exists():
        state = torch.load(ckpt_final, map_location=device)
        print(f"[Pretrained] Loading weights from {ckpt_final}")
    else:
        raise FileNotFoundError(
            f"Neither model_best.pt nor model_final.pt found in {pretrained_ckpt_dir}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Pretrained] Missing keys (expected for dom head): {missing}")
    if unexpected:
        print(f"[Pretrained] Unexpected keys: {unexpected}")

    model.to(device)
    return tokenizer, model, meta


# ------------------ 训练与评估辅助函数 ------------------ #

def schedule_lambda(progress: float) -> float:
    """GRL 系数：从 0 平滑上升到 1"""
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def eval_on_split(model: BertDANN, tokenizer: AutoTokenizer,
                  X_txt, y, max_length: int,
                  device: torch.device, batch_size: int = 64):
    """
    通用评估函数：在给定文本/标签 split 上评估分类精度和 Macro-F1
    可用于 A_val 或 B_val（评估时不走 GRL 和 domain head）
    """
    ds = TextDatasetXY(X_txt, y)
    collate = make_collate_src(tokenizer, max_length)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for input_ids, attn_mask, yb in dl:
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            yb = yb.to(device)

            logits = model(input_ids, attn_mask, lambd=0.0, return_domain=False)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1


# ------------------ main ------------------ #

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_train_jsonl", required=True)
    ap.add_argument("--A_val_jsonl", required=True)
    ap.add_argument("--B_train_jsonl", required=True,
                    help="目标域 B：只使用 sentence1/2，不需要标签")
    ap.add_argument("--B_val_jsonl", required=True,
                    help="目标域 B 验证集：带 gold_label，仅用于 early stopping 和评估")

    ap.add_argument("--pretrained_ckpt_dir", default="ckpt_bert_A_finetune",
                    help="在 A 上用 train_bert_nli_on_A.py 训练好的 BERT checkpoint 目录")
    ap.add_argument("--out_dir", default="ckpt_A2B_bert_dann")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda_dom", type=float, default=0.1,
                    help="domain loss 的权重系数")
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--dp_epsilon", type=float, default=0.0)
    ap.add_argument("--dp_delta", type=float, default=1e-5)
    ap.add_argument("--dp_max_grad_norm", type=float, default=1.0)

    ap.add_argument("--early_stop_patience", type=int, default=3,
                    help="若 B_val acc 连续 N 个 epoch 未提升，则早停")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- 1. 从 A 上预训练 ckpt 加载 BERT + head -------- #
    pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)
    tokenizer, model, meta = build_bert_dann_from_pretrained(
        pretrained_ckpt_dir, device
    )

    # BERT 能支持的最大长度（通常是 512）
    encoder_max_len = getattr(model.encoder.config, "max_position_embeddings", 512)
    # 预训练阶段记录的 max_length（encoder_meta.json 里）+ 当前命令行给的
    meta_max_len = meta.get("max_length", args.max_length)

    # 实际使用的 max_length：三者取最小，确保不会超过 BERT 的 position embedding 上限
    max_length = min(args.max_length, meta_max_len, encoder_max_len)
    print(f"[Config] Using max_length={max_length} "
          f"(encoder_max_len={encoder_max_len}, meta_max_len={meta_max_len})")

    # -------- 2. 加载 A/B 数据 -------- #
    print("[Data] Loading A_train / A_val / B_train / B_val ...")
    Xs_txt, ys = load_jsonl_xy(args.A_train_jsonl)
    Xv_txt, yv = load_jsonl_xy(args.A_val_jsonl)
    Xt_txt = load_jsonl_x(args.B_train_jsonl)   # 目标域 raw 文本列表（无标签）
    Xb_val_txt, yb_val = load_jsonl_xy(args.B_val_jsonl)

    n_B = len(Xt_txt)

    train_src_ds = TextDatasetXY(Xs_txt, ys)
    collate_src = make_collate_src(tokenizer, max_length)

    src_loader = DataLoader(
        train_src_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # DPDataLoader 会忽略这个设置，warning 可以忽略
        collate_fn=collate_src,
    )

    # -------- 3. Optimizer + 可选 DP-SGD -------- #
    # 如果你想只在 head+dom 上做 DP，可以手动将 encoder.requires_grad=False
    # if args.dp_epsilon and args.dp_epsilon > 0:
    #     for p in model.encoder.parameters():
    #         p.requires_grad_(False)
    #     print("[DP-DANN] Freezing encoder parameters (DP on head + domain only)")

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)
    ce_task = nn.CrossEntropyLoss()
    ce_dom = nn.CrossEntropyLoss()

    if args.dp_epsilon and args.dp_epsilon > 0:
        pe = PrivacyEngine()
        model, optimizer, src_loader = pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=src_loader,
            target_epsilon=args.dp_epsilon,
            target_delta=args.dp_delta,
            epochs=args.epochs,
            max_grad_norm=args.dp_max_grad_norm,
        )
        print(
            f"[DP-DANN] Using DP-SGD: eps={args.dp_epsilon}, "
            f"delta={args.dp_delta}, C={args.dp_max_grad_norm}"
        )

    # -------- 4. 训练循环 + B_val 早停 -------- #
    best_A_val_acc = 0.0
    best_A_val_f1 = 0.0
    best_B_val_acc = 0.0
    best_B_val_f1 = 0.0
    best_epoch = 0
    best_state_dict = None
    epochs_no_improve = 0

    train_history = []

    print("[Train] Start BERT-based DANN training ...")

    for ep in range(1, args.epochs + 1):
        model.train()
        n_step = 0

        for input_ids_s, attn_s, yb_s in src_loader:
            # 源域 batch
            input_ids_s = input_ids_s.to(device)
            attn_s = attn_s.to(device)
            yb_s = yb_s.to(device)
            bs_cur = input_ids_s.size(0)

            # 随机从 B 的 raw 文本列表采样一个等长 batch
            idx_t = np.random.randint(0, n_B, size=bs_cur)
            texts_t = [Xt_txt[i] for i in idx_t]
            enc_t = tokenizer(
                texts_t,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids_t = enc_t["input_ids"].to(device)
            attn_t = enc_t["attention_mask"].to(device)

            # 合并成一个 [2B] 的 batch —— 一次 forward，兼容 Opacus
            input_ids_all = torch.cat([input_ids_s, input_ids_t], dim=0)
            attn_all = torch.cat([attn_s, attn_t], dim=0)

            # 训练进度 -> GRL 系数
            progress = (ep - 1 + n_step / len(src_loader)) / args.epochs
            lambd = schedule_lambda(progress)

            # 一次 forward：得到任务 logits 和域 logits
            y_logits_all, d_logits_all = model(
                input_ids_all, attn_all, lambd=lambd, return_domain=True
            )

            # 前半部分是源域的任务 logits
            y_logits_s = y_logits_all[:bs_cur]

            # 域标签：前 B 个为 0 (source)，后 B 个为 1 (target)
            dlab_s = torch.zeros(bs_cur, dtype=torch.long, device=device)
            dlab_t = torch.ones(bs_cur, dtype=torch.long, device=device)
            dlab_all = torch.cat([dlab_s, dlab_t], dim=0)

            # 损失
            loss_task = ce_task(y_logits_s, yb_s)
            loss_dom = ce_dom(d_logits_all, dlab_all)
            loss = loss_task + args.lambda_dom * loss_dom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_step += 1

        # ======== epoch 结束后：在 A_val / B_val 上评估 ======== #
        A_val_acc, A_val_f1 = eval_on_split(
            model, tokenizer, Xv_txt, yv, max_length, device, batch_size=16
        )
        B_val_acc, B_val_f1 = eval_on_split(
            model, tokenizer, Xb_val_txt, yb_val, max_length, device, batch_size=16
        )

        print(
            f"epoch {ep:02d}: "
            f"A-val acc={A_val_acc:.4f}, f1={A_val_f1:.4f} | "
            f"B-val acc={B_val_acc:.4f}, f1={B_val_f1:.4f} | "
            f"lambda≈{lambd:.2f}"
        )

        train_history.append(
            {
                "epoch": ep,
                "A_val_acc": A_val_acc,
                "A_val_f1": A_val_f1,
                "B_val_acc": B_val_acc,
                "B_val_f1": B_val_f1,
                "lambda": lambd,
            }
        )

        # ======== 以 B_val 为准做 early stopping / 选择 best 模型 ======== #
        if B_val_acc > best_B_val_acc:
            best_B_val_acc = B_val_acc
            best_B_val_f1 = B_val_f1
            best_A_val_acc = A_val_acc
            best_A_val_f1 = A_val_f1
            best_epoch = ep
            epochs_no_improve = 0

            # 保存一份当前最优的 state_dict 在内存中
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            print(f"[Best] New best B-val acc={best_B_val_acc:.4f} at epoch {ep}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stop_patience:
                print(
                    f"[EarlyStop] B-val acc has not improved for "
                    f"{args.early_stop_patience} epochs. Stop at epoch {ep}."
                )
                break

    # -------- 5. 保存模型与配置 -------- #
    if best_state_dict is not None:
        # 将最优权重加载回 model，再保存
        model.load_state_dict(best_state_dict, strict=True)
    else:
        # 理论上不会发生，但为了安全起见
        best_epoch = len(train_history)
        best_A_val_acc = train_history[-1]["A_val_acc"]
        best_A_val_f1 = train_history[-1]["A_val_f1"]
        best_B_val_acc = train_history[-1]["B_val_acc"]
        best_B_val_f1 = train_history[-1]["B_val_f1"]

    torch.save(model.state_dict(), str(out_dir / "bert_dann_model.pt"))

    meta_out = meta.copy()
    meta_out["from_pretrained_ckpt_dir"] = str(pretrained_ckpt_dir)
    meta_out["domain_adaptation"] = "bert_dann"
    with open(out_dir / "encoder_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    config = {
        "model_type": "bert_dann",
        "pretrained_ckpt_dir": str(pretrained_ckpt_dir),
        "dp_epsilon": args.dp_epsilon if args.dp_epsilon > 0 else "none",
        "dp_delta": args.dp_delta if args.dp_epsilon > 0 else "none",
        "dp_max_grad_norm": args.dp_max_grad_norm if args.dp_epsilon > 0 else "none",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_dom": args.lambda_dom,
        "max_length": max_length,
        "best_epoch": best_epoch,
        "best_A_val_acc": best_A_val_acc,
        "best_A_val_f1": best_A_val_f1,
        "best_B_val_acc": best_B_val_acc,
        "best_B_val_f1": best_B_val_f1,
        "early_stop_patience": args.early_stop_patience,
        "seed": SEED,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(out_dir / "train_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "A_val_acc", "A_val_f1",
                          "B_val_acc", "B_val_f1", "lambda"]
        )
        writer.writeheader()
        writer.writerows(train_history)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "dann_A2B_results.csv"

    result_row = {
        "experiment_id": str(out_dir),
        "model_type": "bert_dann",
        "source_domain": "A",
        "target_domain": "B",
        "dp_epsilon": args.dp_epsilon if args.dp_epsilon > 0 else "none",
        "dp_delta": args.dp_delta if args.dp_epsilon > 0 else "none",
        "dp_max_grad_norm": args.dp_max_grad_norm if args.dp_epsilon > 0 else "none",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "lambda_dom": args.lambda_dom,
        "max_length": max_length,
        "best_epoch": best_epoch,
        "best_A_val_acc": best_A_val_acc,
        "best_A_val_f1": best_A_val_f1,
        "best_B_val_acc": best_B_val_acc,
        "best_B_val_f1": best_B_val_f1,
        "early_stop_patience": args.early_stop_patience,
    }

    file_exists = results_file.exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    print(f"\n[Done] Saved BERT-DANN model (best on B-val) to {out_dir/'bert_dann_model.pt'}")
    print(f"[Done] Results appended to {results_file}")
    print(
        f"[Summary] best_epoch={best_epoch}, "
        f"A_val_acc={best_A_val_acc:.4f}, B_val_acc={best_B_val_acc:.4f}"
    )
