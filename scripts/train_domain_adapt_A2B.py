#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain Adaptation: BERT-based DANN / DP-DANN

- Backbone: Same BERT encoder + (logistic/MLP) head as train_bert_nli_on_A.py
- Initialization: Load (encoder + head) weights from checkpoint pretrained on Hospital A
- Domain head: MLP + Gradient Reversal Layer (GRL) on sentence embedding
- Training data:
    * Source domain A: Has NLI labels (entailment / contradiction / neutral)
    * Target domain B: No labels (during training), only participates in domain classification
- Validation / Early stop:
    * A_val: Used to monitor source domain performance (optional analysis)
    * B_val: Has labels, but only used for validation and early stopping, not in training
- Optional: Opacus DP-SGD

Input/Output:
  * Input JSONL:
      - --A_train_jsonl:  A_train with gold_label
      - --A_val_jsonl:    A_val with gold_label (only for monitoring classification accuracy on A)
      - --B_train_jsonl:  B_train without labels, only participates in domain classification
      - --B_val_jsonl:    B_val with gold_label, used for early stopping and selecting best B model
  * Read from --pretrained_ckpt_dir:
      - encoder_meta.json
      - model_best.pt (preferred) or model_final.pt
  * Save to --out_dir:
      - bert_dann_model.pt      (checkpoint with best acc on B_val)
      - encoder_meta.json
      - config.json
      - train_history.csv
      - Append one row to results/dann_A2B_results.csv
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

# Constants and random seed

SEED = 2025
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# Data loading functions

def load_jsonl_xy(path):
    """Loads A/B domain data: sentence1, sentence2, gold_label -> (texts, labels)"""
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
    """Loads B_train - just sentence1 + [SEP] + sentence2 (no labels)"""
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
    """Dataset for labeled text (A_train / A_val / B_val)"""

    def __init__(self, texts, y):
        self.texts = texts
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.y[idx]


def make_collate_src(tokenizer, max_length):
    """Returns batch as (input_ids, attention_mask, labels)"""

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding="max_length",      # pad to max_length
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.stack(labels)
        return enc["input_ids"], enc["attention_mask"], labels

    return collate


# BERT model structures (keep aligned with baseline)

def mean_pooling(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling - same as train_bert_nli_on_A.py"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class MLPHead(nn.Module):
    """MLP head - same as train_bert_nli_on_A.py"""

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
    """Simple linear head"""

    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, x):
        return self.fc(x)


# GRL and BERT-DANN model

class GRL(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation"""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class BertDANN(nn.Module):
    """
    BERT-DANN model
      - encoder: pretrained BERT
      - head:    NLI classifier (logistic or MLP)
      - dom:     domain classifier (source=0, target=1)
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
        If return_domain=True: returns (task_logits, domain_logits)
        If return_domain=False: only task_logits (for eval)
        """
        sent = self.encode(input_ids, attention_mask)
        y_logits = self.head(sent)
        if not return_domain:
            return y_logits

        fd = GRL.apply(sent, lambd)
        d_logits = self.dom(fd)
        return y_logits, d_logits


# Build BERT-DANN from pretrained checkpoint

def build_bert_dann_from_pretrained(pretrained_ckpt_dir: Path,
                                    device: torch.device):
    """
    Loads checkpoint from train_bert_nli_on_A.py:
      - encoder_meta.json
      - model_best.pt (preferred) or model_final.pt
    Builds BertDANN (inherits encoder+head weights, dom head is randomly init)
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

    # Load pretrained weights from A (encoder + head only, dom head stays random)
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


# Training and eval helpers

def schedule_lambda(progress: float) -> float:
    """GRL lambda schedule: smoothly goes from 0 to 1"""
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def eval_on_split(model: BertDANN, tokenizer: AutoTokenizer,
                  X_txt, y, max_length: int,
                  device: torch.device, batch_size: int = 64):
    """
    Eval function for any labeled split (A_val or B_val)
    Returns acc and macro-F1. Note: eval doesn't use GRL/domain head
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
                    help="Target domain B: only use sentence1/2, no labels needed")
    ap.add_argument("--B_val_jsonl", required=True,
                    help="Target domain B validation set: with gold_label, only for early stopping and evaluation")

    ap.add_argument("--pretrained_ckpt_dir", default="ckpt_bert_A_finetune",
                    help="BERT checkpoint directory trained on A using train_bert_nli_on_A.py")
    ap.add_argument("--out_dir", default="ckpt_A2B_bert_dann")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda_dom", type=float, default=0.1,
                    help="Weight coefficient for domain loss")
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--dp_epsilon", type=float, default=0.0)
    ap.add_argument("--dp_delta", type=float, default=1e-5)
    ap.add_argument("--dp_max_grad_norm", type=float, default=1.0)

    ap.add_argument("--early_stop_patience", type=int, default=3,
                    help="Early stop if B_val acc does not improve for N consecutive epochs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pretrained BERT + head from A
    pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)
    tokenizer, model, meta = build_bert_dann_from_pretrained(
        pretrained_ckpt_dir, device
    )

    # Get max length constraints
    encoder_max_len = getattr(model.encoder.config, "max_position_embeddings", 512)
    meta_max_len = meta.get("max_length", args.max_length)
    # Use the minimum to avoid exceeding BERT's position embedding limit
    max_length = min(args.max_length, meta_max_len, encoder_max_len)
    print(f"[Config] Using max_length={max_length} "
          f"(encoder_max_len={encoder_max_len}, meta_max_len={meta_max_len})")

    # 2. Load A/B data
    print("[Data] Loading A_train / A_val / B_train / B_val ...")
    Xs_txt, ys = load_jsonl_xy(args.A_train_jsonl)
    Xv_txt, yv = load_jsonl_xy(args.A_val_jsonl)
    Xt_txt = load_jsonl_x(args.B_train_jsonl)   # B_train: just text, no labels
    Xb_val_txt, yb_val = load_jsonl_xy(args.B_val_jsonl)

    n_B = len(Xt_txt)

    train_src_ds = TextDatasetXY(Xs_txt, ys)
    collate_src = make_collate_src(tokenizer, max_length)

    src_loader = DataLoader(
        train_src_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # DPDataLoader ignores this anyway
        collate_fn=collate_src,
    )

    # 3. Setup optimizer (optional DP-SGD)
    # Note: can freeze encoder if you only want DP on head+dom
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

    # 4. Training loop (early stop on B_val)
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
            # Source domain batch
            input_ids_s = input_ids_s.to(device)
            attn_s = attn_s.to(device)
            yb_s = yb_s.to(device)
            bs_cur = input_ids_s.size(0)

            # Sample matching batch from B (no labels)
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

            # Concatenate into [2B] batch for single forward (Opacus compatible)
            input_ids_all = torch.cat([input_ids_s, input_ids_t], dim=0)
            attn_all = torch.cat([attn_s, attn_t], dim=0)

            # Compute GRL lambda based on training progress
            progress = (ep - 1 + n_step / len(src_loader)) / args.epochs
            lambd = schedule_lambda(progress)

            # Forward pass
            y_logits_all, d_logits_all = model(
                input_ids_all, attn_all, lambd=lambd, return_domain=True
            )

            # Split task logits (first half is source domain)
            y_logits_s = y_logits_all[:bs_cur]

            # Domain labels: source=0, target=1
            dlab_s = torch.zeros(bs_cur, dtype=torch.long, device=device)
            dlab_t = torch.ones(bs_cur, dtype=torch.long, device=device)
            dlab_all = torch.cat([dlab_s, dlab_t], dim=0)

            # Compute losses
            loss_task = ce_task(y_logits_s, yb_s)
            loss_dom = ce_dom(d_logits_all, dlab_all)
            loss = loss_task + args.lambda_dom * loss_dom

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_step += 1

        # Eval on A_val / B_val after each epoch
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

        # Early stopping based on B_val acc
        if B_val_acc > best_B_val_acc:
            best_B_val_acc = B_val_acc
            best_B_val_f1 = B_val_f1
            best_A_val_acc = A_val_acc
            best_A_val_f1 = A_val_f1
            best_epoch = ep
            epochs_no_improve = 0

            # Keep best model in memory
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

    # 5. Save best model and config
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=True)
    else:
        # Fallback (shouldn't happen)
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
