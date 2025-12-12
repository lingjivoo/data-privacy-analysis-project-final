#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain Adaptation: BERT-based DANN / DP-DANN

- Backbone: Same BERT encoder + (logistic/MLP) head as train_bert_nli_on_A.py
- Initialization: Load (encoder + head) weights from checkpoint pretrained on Hospital A
- Domain head: MLP + Gradient Reversal Layer (GRL) on sentence embedding
- Training data:
    * Source domain A: Has NLI labels (entailment / contradiction / neutral)
    * Target domain B: No labels, only participates in domain classification
- Optional: Opacus DP-SGD

Input/Output:
  * Input JSONL:
      - --A_train_jsonl:  A_train with gold_label
      - --A_val_jsonl:    A_val with gold_label (only for monitoring classification accuracy on A)
      - --B_train_jsonl:  B_train without labels, only use sentence1/sentence2
      - --B_val_jsonl:    B_val with gold_label, only for validation and recording best B model
  * Read from --pretrained_ckpt_dir:
      - encoder_meta.json
      - model_best.pt (preferred) or model_final.pt
  * Save to --out_dir:
      - bert_dann_model.pt   (checkpoint with best acc on B_val)
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

# ------------------ Constants and random seed ------------------ #

SEED = 2025
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ------------------ Data loading ------------------ #

def load_jsonl_xy(path):
    """Load A/B domain: sentence1, sentence2, gold_label -> (texts, labels)"""
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
    """Load B_train: only take sentence1 + [SEP] + sentence2"""
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
    """Labeled text Dataset for A_val / B_val (eval, does not go through DP)"""

    def __init__(self, texts, y):
        self.texts = texts
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.y[idx]


def make_collate_src(tokenizer, max_length):
    """Collate used for eval: (input_ids, attention_mask, labels)"""

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.stack(labels)
        return enc["input_ids"], enc["attention_mask"], labels

    return collate


# ====== Training Dataset for Opacus (__getitem__ directly returns Tensor) ====== #

class BertSrcDataset(Dataset):
    """
    Source domain A: Complete tokenizer in __getitem__,
    return (input_ids [L], attention_mask [L], label), all as Tensor.
    This way PrivacyEngine-generated DPDataLoader won't encounter strings.
    """
    def __init__(self, texts, y, tokenizer, max_length):
        self.texts = texts
        self.y = torch.tensor(y, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)            # [L]
        attention_mask = enc["attention_mask"].squeeze(0)  # [L]
        label = self.y[idx]                                # scalar long
        return input_ids, attention_mask, label


# ------------------ BERT structure (aligned with baseline) ------------------ #

def mean_pooling(last_hidden_state: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """Same mean pooling implementation as train_bert_nli_on_A.py"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class MLPHead(nn.Module):
    """Multi-layer MLP head same as train_bert_nli_on_A.py"""

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
    """Single-layer linear classification head"""

    def __init__(self, d, c):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, x):
        return self.fc(x)


# ------------------ GRL + BERT-DANN model ------------------ #

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
      - encoder: Pretrained BERT encoder
      - head:    NLI classification head (logistic / MLP)
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
        return_domain=True: Return (task_logits, domain_logits)
        return_domain=False: Only return task_logits (for evaluation)
        """
        sent = self.encode(input_ids, attention_mask)
        y_logits = self.head(sent)
        if not return_domain:
            return y_logits

        fd = GRL.apply(sent, lambd)
        d_logits = self.dom(fd)
        return y_logits, d_logits


# ------------------ Build BERT-DANN from pretrained checkpoint on A ------------------ #

def build_bert_dann_from_pretrained(pretrained_ckpt_dir: Path,
                                    device: torch.device):
    """
    Load from ckpt_dir trained by train_bert_nli_on_A.py:
      - encoder_meta.json
      - model_best.pt (preferred) or model_final.pt
    Build BertDANN model (encoder+head weights inherited, dom head randomly initialized).
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

    # Load A domain pretrained weights: only cover encoder + head, dom head randomly initialized
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


# ------------------ Training and evaluation helper functions ------------------ #

def schedule_lambda(progress: float) -> float:
    """GRL coefficient: smoothly increase from 0 to 1"""
    return 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0


def eval_on_split(model: BertDANN, tokenizer: AutoTokenizer,
                  X_txt, y, max_length: int,
                  device: torch.device, batch_size: int = 64):
    """General evaluation function: evaluate acc / macro-F1 on any labeled split"""
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


class HeadDom(nn.Module):
    def __init__(self, head, dom):
        super().__init__()
        self.head = head
        self.dom = dom


# ------------------ main ------------------ #

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--A_train_jsonl", required=True)
    ap.add_argument("--A_val_jsonl", required=True)
    ap.add_argument("--B_train_jsonl", required=True,
                    help="Target domain B: only use sentence1/2, no labels needed")
    ap.add_argument("--B_val_jsonl", required=True,
                    help="Target domain B validation set: with gold_label, only for evaluation and selecting best model")

    ap.add_argument("--pretrained_ckpt_dir", default="ckpt_bert_A_finetune",
                    help="BERT checkpoint directory trained on A using train_bert_nli_on_A.py")
    ap.add_argument("--out_dir", default="ckpt_A2B_bert_dann_dp")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lambda_dom", type=float, default=0.1,
                    help="Weight coefficient for domain loss")
    ap.add_argument("--max_length", type=int, default=512)

    ap.add_argument("--dp_epsilon", type=float, default=0.0)
    ap.add_argument("--dp_delta", type=float, default=1e-5)
    ap.add_argument("--dp_max_grad_norm", type=float, default=1.0)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- 1. Load BERT + head from pretrained ckpt on A -------- #
    pretrained_ckpt_dir = Path(args.pretrained_ckpt_dir)
    tokenizer, model, meta = build_bert_dann_from_pretrained(
        pretrained_ckpt_dir, device
    )

    # Maximum length BERT can support (usually 512)
    encoder_max_len = getattr(model.encoder.config, "max_position_embeddings", 512)
    # max_length recorded during pretraining (in encoder_meta.json) + current command line value
    meta_max_len = meta.get("max_length", args.max_length)

    # Actual max_length used: take minimum of three to ensure it doesn't exceed BERT's position embedding limit
    max_length = min(args.max_length, meta_max_len, encoder_max_len)
    print(f"[Config] Using max_length={max_length} "
          f"(encoder_max_len={encoder_max_len}, meta_max_len={meta_max_len})")

    # -------- 2. Load A/B data -------- #
    print("[Data] Loading A_train / A_val / B_train / B_val ...")
    Xs_txt, ys = load_jsonl_xy(args.A_train_jsonl)
    Xv_txt, yv = load_jsonl_xy(args.A_val_jsonl)
    Xt_txt = load_jsonl_x(args.B_train_jsonl)   # Target domain raw text list (no labels)
    Xb_val_txt, yb_val = load_jsonl_xy(args.B_val_jsonl)

    n_B = len(Xt_txt)

    # For training: BertSrcDataset (__getitem__ directly returns Tensor, for Opacus use)
    train_src_ds = BertSrcDataset(Xs_txt, ys, tokenizer, max_length)
    src_loader = DataLoader(
        train_src_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,  # DPDataLoader will ignore this setting, warning can be ignored
    )

    # -------- 3. Optimizer + optional DP-SGD -------- #
    # When using DP, only apply noise to head + domain, encoder uses regular optimizer
    enc_params = list(model.encoder.parameters())
    hd_params = list(model.head.parameters()) + list(model.dom.parameters())
    optimizer_enc = torch.optim.AdamW(enc_params, lr=args.lr)  # Regular optimizer
    optimizer_hd = torch.optim.AdamW(hd_params, lr=args.lr)    # For DP

    ce_task = nn.CrossEntropyLoss()
    ce_dom = nn.CrossEntropyLoss()
    hd_module = HeadDom(model.head, model.dom)

    if args.dp_epsilon and args.dp_epsilon > 0:
        pe = PrivacyEngine()
        hd_module, optimizer_hd, src_loader = pe.make_private_with_epsilon(
            module=hd_module,
            optimizer=optimizer_hd,
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

    # -------- 4. Training loop (A/B merged batch, single forward) -------- #
    best_A_val_acc = 0.0
    best_A_val_f1 = 0.0
    best_B_val_acc = 0.0
    best_B_val_f1 = 0.0
    best_epoch = 0
    best_state_dict = None

    train_history = []

    print("[Train] Start BERT-based DANN (DP) training ...")

    for ep in range(1, args.epochs + 1):
        model.train()
        n_step = 0

        for input_ids_s, attn_s, yb_s in src_loader:
            # DPDataLoader (Poisson sampling) may give empty batch, skip directly
            if input_ids_s.numel() == 0:
                continue

            # Source domain batch
            input_ids_s = input_ids_s.to(device)
            attn_s = attn_s.to(device)
            yb_s = yb_s.to(device)
            bs_cur = input_ids_s.size(0)

            # Randomly sample an equal-length batch from B's raw text list
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

            # Merge into a [2B] batch - single forward pass, compatible with Opacus
            input_ids_all = torch.cat([input_ids_s, input_ids_t], dim=0)
            attn_all = torch.cat([attn_s, attn_t], dim=0)

            # Progress -> GRL coefficient
            progress = (ep - 1 + n_step / len(src_loader)) / args.epochs
            lambd = schedule_lambda(progress)

            # Single forward pass: get task logits and domain logits
            y_logits_all, d_logits_all = model(
                input_ids_all, attn_all, lambd=lambd, return_domain=True
            )

            # First half is source domain task logits
            y_logits_s = y_logits_all[:bs_cur]

            # Domain labels: first B are 0 (source), last B are 1 (target)
            dlab_s = torch.zeros(bs_cur, dtype=torch.long, device=device)
            dlab_t = torch.ones(bs_cur, dtype=torch.long, device=device)
            dlab_all = torch.cat([dlab_s, dlab_t], dim=0)

            # Loss
            loss_task = ce_task(y_logits_s, yb_s)
            loss_dom = ce_dom(d_logits_all, dlab_all)
            loss = loss_task + args.lambda_dom * loss_dom

            optimizer_enc.zero_grad()
            optimizer_hd.zero_grad()

            loss.backward()
            optimizer_enc.step()
            optimizer_hd.step()
            n_step += 1

        # ======== After epoch ends: evaluate on A_val / B_val ======== #
        A_val_acc, A_val_f1 = eval_on_split(
            model, tokenizer, Xv_txt, yv, max_length, device, batch_size=64
        )
        B_val_acc, B_val_f1 = eval_on_split(
            model, tokenizer, Xb_val_txt, yb_val, max_length, device, batch_size=64
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

        # Use B_val acc as standard, record best model (but no early stop, ensure DP-ε aligns with specified epochs)
        if B_val_acc > best_B_val_acc:
            best_B_val_acc = B_val_acc
            best_B_val_f1 = B_val_f1
            best_A_val_acc = A_val_acc
            best_A_val_f1 = A_val_f1
            best_epoch = ep
            # Copy current model weights
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            print(f"[Best] New best B-val acc={best_B_val_acc:.4f} at epoch {ep}")

    # -------- 5. Save model and configuration -------- #
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict, strict=True)
    else:
        # Should not happen in theory, but as a fallback
        last = train_history[-1]
        best_epoch = last["epoch"]
        best_A_val_acc = last["A_val_acc"]
        best_A_val_f1 = last["A_val_f1"]
        best_B_val_acc = last["B_val_acc"]
        best_B_val_f1 = last["B_val_f1"]

    torch.save(model.state_dict(), str(out_dir / "bert_dann_model.pt"))

    meta_out = meta.copy()
    meta_out["from_pretrained_ckpt_dir"] = str(pretrained_ckpt_dir)
    meta_out["domain_adaptation"] = "bert_dann_dp"
    with open(out_dir / "encoder_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2, ensure_ascii=False)

    config = {
        "model_type": "bert_dann_dp",
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
        "seed": SEED,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(out_dir / "train_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "A_val_acc", "A_val_f1",
                "B_val_acc", "B_val_f1",
                "lambda",
            ],
        )
        writer.writeheader()
        writer.writerows(train_history)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "dann_A2B_results.csv"

    result_row = {
        "experiment_id": str(out_dir),
        "model_type": "bert_dann_dp",
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
    }

    file_exists = results_file.exists()
    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_row)

    print(f"\n[Done] Saved DP BERT-DANN model (best on B-val) to {out_dir/'bert_dann_model.pt'}")
    print(f"[Done] Results appended to {results_file}")
    print(
        f"[Summary] best_epoch={best_epoch}, "
        f"A_val_acc={best_A_val_acc:.4f}, B_val_acc={best_B_val_acc:.4f}"
    )
