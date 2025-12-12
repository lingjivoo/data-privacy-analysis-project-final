#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT/BioBERT sentence-embedding baseline for NLI (Hospital A).

Two modes:
1) Frozen encoder + trained head (default, with cached embeddings).
2) Joint training (finetune encoder + head) when --finetune_encoder is set.

Encoder: HuggingFace transformer
Head: PyTorch MLP / logistic classifier (CE loss)
Optional: Opacus DP-SGD (set --dp_epsilon>0 to enable)
Saves: encoder_meta.json, *.pt models, and (in frozen mode) cached *.npy embeddings
"""

import json
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from opacus import PrivacyEngine

LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(p: str) -> Tuple[List[str], np.ndarray]:
    """Load NLI jsonl: sentence1, sentence2, gold_label."""
    X, y = [], []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            s1 = o.get("sentence1", "")
            s2 = o.get("sentence2", "")
            s = (s1 + " [SEP] " + s2).strip()
            X.append(s)
            y.append(LABEL2ID[o["gold_label"]])
    return X, np.array(y, dtype=np.int64)


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-masked tokens."""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, T, 1]
    summed = (last_hidden_state * mask).sum(dim=1)  # [B, H]
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)  # [B, 1]
    return summed / counts


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
    pooling: str = "mean",
) -> np.ndarray:
    """Run encoder once to get fixed sentence embeddings (frozen mode)."""
    model.eval()
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc)
        if pooling == "mean":
            sent = mean_pooling(out.last_hidden_state, enc["attention_mask"])
        elif pooling == "cls":
            sent = out.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
        embs.append(sent.cpu().numpy())
    return np.vstack(embs)


def maybe_cache_embeddings(
    cache_path: Path,
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    """Load embeddings from cache if exists; otherwise compute & save."""
    if cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    print(f"Encoding texts and caching to {cache_path}")
    arr = encode_texts(
        texts,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length,
        pooling=pooling,
    )
    np.save(cache_path, arr)
    return arr


class TensorDatasetXY(Dataset):
    """For frozen encoder mode: X is already a float32 matrix."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TextDataset(Dataset):
    """For finetune mode: store raw texts + labels."""

    def __init__(self, texts: List[str], y: np.ndarray):
        self.texts = texts
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.y[idx]


def make_collate_fn(tokenizer: AutoTokenizer, max_length: int):
    """Collate raw texts into token ids + attention mask + labels."""

    def collate(batch):
        texts, labels = zip(*batch)  # tuple(str), tuple(tensor)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        labels = torch.stack(labels)  # [B]
        return enc["input_ids"], enc["attention_mask"], labels

    return collate


class MLPHead(nn.Module):
    """Multi-layer perceptron classification head."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims=None,
        dropout: float = 0.3,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            if use_batchnorm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Logistic(nn.Module):
    """Single-layer linear classifier."""

    def __init__(self, d: int, c: int):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class EncHead(nn.Module):
    """Wrapper for encoder + head (used in finetune mode)."""

    def __init__(self, encoder: AutoModel, head: nn.Module, pooling: str = "mean"):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            sent = mean_pooling(out.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            sent = out.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        return self.head(sent)


def evaluate_frozen(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Evaluate when encoder is frozen and loader yields (X, y)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return acc, avg_loss


def evaluate_finetune(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    """Evaluate when encoder is finetuned and loader yields (input_ids, attention_mask, y)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = 0
    total_loss = 0.0

    with torch.no_grad():
        for input_ids, attention_mask, yb in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            yb = yb.to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, yb)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.size(0)

    avg_loss = total_loss / max(len(loader), 1)
    acc = correct / max(total, 1)
    return acc, avg_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl", required=True)
    ap.add_argument("--out_dir", default="ckpt_bert_A")

    # Encoder
    ap.add_argument(
        "--encoder_name",
        default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        help="Any HuggingFace encoder",
    )
    ap.add_argument("--pooling", default="mean", choices=["mean", "cls"])
    ap.add_argument("--max_length", type=int, default=512)

    # Head / architecture
    ap.add_argument("--head_type", default="logistic", choices=["logistic", "mlp"])
    ap.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer dims for MLP head",
    )
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument(
        "--use_batchnorm",
        action="store_true",
        default=True,
        help="Use LayerNorm in MLP head",
    )

    # Training
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=32, help="Train batch size")
    ap.add_argument(
        "--enc_batch_size",
        type=int,
        default=64,
        help="Batch size for pre-encoding (frozen mode)",
    )
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--optimizer", default="adamw", choices=["sgd", "adam", "adamw"])
    ap.add_argument("--seed", type=int, default=42)

    # Differential Privacy
    ap.add_argument("--dp_epsilon", type=float, default=0.0, help=">0 to enable DP-SGD")
    ap.add_argument("--dp_delta", type=float, default=1e-5)
    ap.add_argument("--dp_max_grad_norm", type=float, default=1.0)

    # Finetune encoder?
    ap.add_argument(
        "--finetune_encoder",
        action="store_true",
        help="Jointly train encoder + head instead of using frozen embeddings",
    )

    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== Load data =====
    print("Loading data...")
    Xtr_txt, ytr = load_jsonl(args.train_jsonl)
    Xv_txt, yv = load_jsonl(args.val_jsonl)
    print(f"Train size: {len(Xtr_txt)}, Val size: {len(Xv_txt)}")

    # ===== Load encoder & tokenizer =====
    print(f"Loading encoder: {args.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, use_fast=True)
    encoder = AutoModel.from_pretrained(args.encoder_name, use_safetensors=True).to(device)

    # ===== Build model / dataloaders for two modes =====
    c = 3  # num classes

    if not args.finetune_encoder:
        print("Mode: frozen encoder + trained head")

        # Precompute & cache embeddings
        tr_cache = out_dir / f"train_emb_{args.pooling}_L{args.max_length}.npy"
        va_cache = out_dir / f"val_emb_{args.pooling}_L{args.max_length}.npy"
        Xtr = maybe_cache_embeddings(
            tr_cache,
            Xtr_txt,
            tokenizer,
            encoder,
            device,
            args.enc_batch_size,
            args.max_length,
            args.pooling,
        )
        Xv = maybe_cache_embeddings(
            va_cache,
            Xv_txt,
            tokenizer,
            encoder,
            device,
            args.enc_batch_size,
            args.max_length,
            args.pooling,
        )

        d = Xtr.shape[1]
        print(f"Embedding dimension: {d}, Num classes: {c}")

        if args.head_type == "logistic":
            head = Logistic(d, c).to(device)
            print("Using logistic regression head")
        else:
            head = MLPHead(
                input_dim=d,
                num_classes=c,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
                use_batchnorm=args.use_batchnorm,
            ).to(device)
            print(
                f"Using MLP head with hidden dims={args.hidden_dims}, "
                f"dropout={args.dropout}, use_batchnorm={args.use_batchnorm}"
            )

        model = head  # in frozen mode, model == head
        print(f"Total trainable parameters (head only): {sum(p.numel() for p in model.parameters()):,}")

        train_loader = DataLoader(
            TensorDatasetXY(Xtr, ytr),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            TensorDatasetXY(Xv, yv),
            batch_size=1024,
            shuffle=False,
        )

    else:
        print("Mode: finetune encoder + head (joint training)")

        # Determine encoder output dimension by a dummy forward
        with torch.no_grad():
            enc_sample = tokenizer(
                [Xtr_txt[0]],
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)
            out = encoder(**enc_sample)
            if args.pooling == "mean":
                sent = mean_pooling(out.last_hidden_state, enc_sample["attention_mask"])
            else:
                sent = out.last_hidden_state[:, 0, :]
            d = sent.size(1)

        print(f"Encoder hidden dim: {d}, Num classes: {c}")

        if args.head_type == "logistic":
            head = Logistic(d, c)
            print("Using logistic regression head")
        else:
            head = MLPHead(
                input_dim=d,
                num_classes=c,
                hidden_dims=args.hidden_dims,
                dropout=args.dropout,
                use_batchnorm=args.use_batchnorm,
            )
            print(
                f"Using MLP head with hidden dims={args.hidden_dims}, "
                f"dropout={args.dropout}, use_batchnorm={args.use_batchnorm}"
            )

        model = EncHead(encoder, head, pooling=args.pooling).to(device)
        print(f"Total trainable parameters (encoder + head): {sum(p.numel() for p in model.parameters()):,}")

        # Datasets + loaders on raw text
        collate = make_collate_fn(tokenizer, args.max_length)
        train_ds = TextDataset(Xtr_txt, ytr)
        val_ds = TextDataset(Xv_txt, yv)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=512,
            shuffle=False,
            collate_fn=collate,
        )

    # ===== Optimizer =====
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:  # adamw
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    criterion = nn.CrossEntropyLoss()

    # ===== Optional DP-SGD =====
    if args.dp_epsilon and args.dp_epsilon > 0:
        pe = PrivacyEngine()
        model, optimizer, train_loader = pe.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            target_epsilon=args.dp_epsilon,
            target_delta=args.dp_delta,
            epochs=args.epochs,
            max_grad_norm=args.dp_max_grad_norm,
        )
        print(
            f"[DP] Using DP-SGD with eps={args.dp_epsilon}, "
            f"delta={args.dp_delta}, C={args.dp_max_grad_norm}"
        )

    # ===== Training loop =====
    print(f"\nTraining for {args.epochs} epochs...")
    best_val_acc = 0.0
    best_epoch = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            if args.finetune_encoder:
                input_ids, attention_mask, yb = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                yb = yb.to(device)
                logits = model(input_ids, attention_mask)
            else:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)

            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation
        if args.finetune_encoder:
            val_acc, val_loss = evaluate_finetune(model, val_loader, device)
        else:
            val_acc, val_loss = evaluate_frozen(model, val_loader, device)

        print(
            f"Epoch {ep:3d}: train_loss={avg_train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}",
            end="",
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            print(" <- BEST")

            if args.finetune_encoder:
                torch.save(model.state_dict(), str(out_dir / "model_best.pt"))
            else:
                torch.save(model.state_dict(), str(out_dir / "head_best.pt"))
        else:
            print()

    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # ===== Save artifacts =====
    meta = {
        "encoder_name": args.encoder_name,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "hidden_size": int(d),
        "label2id": LABEL2ID,
        "head_type": args.head_type,
        "hidden_dims": args.hidden_dims if args.head_type == "mlp" else None,
        "dropout": args.dropout if args.head_type == "mlp" else None,
        "use_batchnorm": args.use_batchnorm if args.head_type == "mlp" else None,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "finetune_encoder": bool(args.finetune_encoder),
        "dp_epsilon": float(args.dp_epsilon),
        "dp_delta": float(args.dp_delta),
        "dp_max_grad_norm": float(args.dp_max_grad_norm),
    }

    with open(out_dir / "encoder_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.finetune_encoder:
        torch.save(model.state_dict(), str(out_dir / "model_final.pt"))
        print(f"Saved finetuned model to {out_dir/'model_final.pt'} and best to model_best.pt")
    else:
        torch.save(model.state_dict(), str(out_dir / "head_final.pt"))
        print(f"Saved head to {out_dir/'head_final.pt'} and best to head_best.pt")

    if not args.finetune_encoder:
        print("Cached embeddings:", tr_cache.name, va_cache.name)


if __name__ == "__main__":
    main()
