
---

# Differential Privacy for Domain Adaptation on Medical Data

This repository is a reference implementation of the **Differential Privacy for Domain Adaptation on Medical Data** project, which systematically studies:

> Under the premise of **domain shift (Hospital A → Hospital B)**,
> when using **BERT + Domain Adaptation (DANN) + Differentially Private Training (DP-SGD)**,
> how do the model's **utility** and **privacy protection capabilities** change?

The entire experiment can be executed in one go through a complete pipeline script (e.g., `run_dp_mednli_pipeline.sh` in the project root) for **Phase I–III + Utility & Privacy Evaluation + Plotting**.

---

## 1. Problem Definition & Research Questions

We consider two healthcare institutions with distribution differences:

* **Source Domain (Hospital A)**: Labeled NLI data for training.
* **Target Domain (Hospital B)**: Synthesized from A, with controllable domain shifts in terminology/style/format, used for unlabeled domain adaptation and testing.

Task: Given two clinical texts (e.g., premise/hypothesis), predict their relationship:

> `entailment` / `contradiction` / `neutral`

At the same time, we need to:

1. Maintain as high classification performance as possible on the B-domain test set;
2. Comply with **Differential Privacy (DP)** during training, limiting memory and reproduction of individual samples (including synthetic PHI);
3. Analyze **cross-domain privacy leakage**: Attack A and observe whether B's output "carries out" A's private information.

We focus on the following research questions:

* **RQ1 (DP + Domain Adaptation)**
  When adding **DANN domain adaptation** to BERT and then applying DP-SGD (DP-DANN), how does the impact on B-domain performance differ compared to DP-BERT alone?

* **RQ2 (Privacy on A & Cross-Domain Leakage)**

  * How does the membership inference attack (MIA) success rate on **A-domain training data** change under no DP / DP-BERT / DP-DANN?
  * If only observing **B-domain output**, can attackers "indirectly" recover A-domain synthetic PHI (i.e., "attack A, observe B")? Can DP suppress this cross-domain leakage?

* **RQ3 (Utility–Privacy Trade-off)**
  As the DP budget ε changes,

  * How does B-domain utility (Acc/F1) change?
  * How does the privacy attack success rate change?
  * What systematic impact does **with/without domain adaptation (DANN vs baseline)** have on the shape of the trade-off curve?

---

## 2. Experimental Phases & Overall Pipeline

The complete experimental pipeline is divided into three phases (Phase I–III), corresponding to the three main experimental blocks in the final project report.
The accompanying bash pipeline script (e.g., `run_dp_mednli_pipeline.sh`) executes in order:

1. **Phase I – Privacy-Aware Data Construction**
2. **Phase II – BERT + DP Training on Source Domain A (BERT + DP-SGD on A)**
3. **Phase III – A→B Domain Adaptation + DP (DANN & DP-DANN)**

Then uniformly perform:

* **Utility Evaluation**: `eval_on_B.py`, testing baseline / DP-BERT / DANN / DP-DANN on B-test.
* **Privacy Evaluation**: `attack_membership_inference.py` (MIA), `scan_phi_like.py` (PHI-like scanning).
* **Trade-off Plotting**: `plot_tradeoff.py` outputs utility–privacy curves and tables.

---

## 3. Repository Layout

Consistent with the complete pipeline script, the main structure is as follows:

```bash
dp-mednli/
├── corpus_A/                     # Hospital A data: raw + JSONL + synthetic PHI
│   ├── train.txt                 # Original MedNLI-style training file (optional)
│   ├── A_train_leaky.jsonl       # Standardized & synthetic PHI-injected A-train
│   ├── A_val_leaky.jsonl         # A-validation
│   └── A_test_leaky.jsonl        # A-test
├── corpus_B/                     # Hospital B data: synthesized from A
│   ├── shift_config.json         # A→B terminology/style shift configuration
│   ├── B_train.jsonl
│   ├── B_val.jsonl
│   ├── B_test.jsonl
│   └── B_test.clean.jsonl        # Lightly cleaned / quality-checked B-test
├── scripts/
│   # Phase I: Data Construction
│   ├── convert_to_standard_jsonl.py   # train.txt → A_*_leaky.jsonl
│   ├── make_synthetic_phi_A.py        # Inject synthetic PHI into A, generate leak labels
│   ├── hospitalB_shift_pipeline.py    # A → B domain mapping, safe perturbation according to shift_config
│
│   # Phase II: BERT Baseline + DP Training (Source Domain A)
│   ├── train_bert_nli_on_A.py         # Fine-tune BERT on A, supports DP-SGD
│
│   # Phase III: A→B Domain Adaptation (DANN & DP-DANN)
│   ├── train_domain_adapt_A2B.py      # Non-DP BERT-DANN
│   ├── train_domain_adapt_A2B_DP.py   # DP-SGD version of BERT-DANN (DP-DANN)
│
│   # Evaluation & Privacy Attacks
│   ├── eval_on_B.py                   # Evaluate baseline / DANN models on B-test
│   ├── attack_membership_inference.py # Membership Inference Attack (MIA)
│   ├── scan_phi_like.py               # Scan for PHI-like fragments in model output / data
│   └── plot_tradeoff.py               # Aggregate results and plot utility–privacy curves
│
├── ckpt_bert_A_finetune/              # Phase II: BERT baseline (no DP)
├── ckpt_bert_A_finetune_dp_eps*/      # Phase II: DP-BERT (different ε)
├── ckpt_A2B_dann/                     # Phase III: Non-DP DANN
├── ckpt_A2B_dp_eps*/                  # Phase III: DP-DANN (different ε)
├── results/                           # Metrics CSV / attack results / plotting outputs
└── run_dp_mednli_pipeline.sh          # (Example) Complete pipeline script
```

---

## 4. Phase I – Privacy-Aware Data Construction

### 4.1 Standardize Hospital A Data & Synthesize PHI

First step: Convert original MedNLI-style files (`train.txt`, etc.) into standard JSONL format, one sample per line, for example:

```json
{
  "sentence1": "...",
  "sentence2": "...",
  "gold_label": "entailment|contradiction|neutral",
  "phi_text": "synthetic PHI span or empty",
  "leak_flag": 1
}
```

Example commands (already in the pipeline script):

```bash
# (1) train.txt → A_train_leaky.jsonl (if corpus_A/train.txt exists)
python scripts/convert_to_standard_jsonl.py \
  --mode train_txt \
  --input  corpus_A/train.txt \
  --output corpus_A/A_train_leaky.jsonl

# (2) Inject synthetic PHI (Faker) and add leak labels
python scripts/make_synthetic_phi_A.py \
  --in_jsonl  corpus_A/A_train_leaky.jsonl \
  --out_jsonl corpus_A/A_train_leaky.jsonl \
  --leak_prob 0.35 \
  --hospital_name "Hospital A"
```

Optional sanity check:

```bash
python - <<'PY'
import json, collections
p = "corpus_A/A_train_leaky.jsonl"
c = collections.Counter(); n = leaks = 0
for l in open(p, encoding="utf-8"):
    o = json.loads(l); n += 1
    c[o["gold_label"]] += 1
    leaks += o.get("leak_flag", 0)
print("n=", n, "label_dist=", c, "num_leaks=", leaks)
PY
```

### 4.2 Synthesize Hospital B from A (Controlled Domain Shift)

We control the terminology mapping, abbreviation style, unit changes, etc. from A→B through the configuration file `corpus_B/shift_config.json`, while ensuring:

* Replacements are performed under **word boundary / regex constraints**, without breaking numbers/decimal points;
* No new PHI is introduced, only medical expression style and terminology are changed;
* Consistent transformation for train/val/test splits is supported.

Example commands (used in the pipeline script):

```bash
for split in train val test; do
  python scripts/hospitalB_shift_pipeline.py \
    --in_jsonl  corpus_A/A_${split}_leaky.jsonl \
    --out_jsonl corpus_B/B_${split}.jsonl \
    --config    corpus_B/shift_config.json
done
```

If necessary, further cleaning can be performed on `B_*` to produce `B_test.clean.jsonl`, etc., for final evaluation.

---

## 5. Phase II – BERT on A: Baseline & DP Training

This phase fine-tunes BERT on **Hospital A** and adds DP-SGD under multiple ε values, corresponding to "DP-BERT Baseline on Source Domain" in the paper.

### 5.1 Model & Training Settings

* **Encoder**: BERT / BioBERT (specific name specified by defaults or parameters in `train_bert_nli_on_A.py`), supports `--finetune_encoder` to control whether to update the encoder.
* **Head**: NLI classification head (MLP + softmax), outputs 3 classes.
* **Optimizer**: AdamW or equivalent settings.
* **DP Mechanism**: Uses Opacus DP-SGD, training under given `(ε, δ, max_grad_norm)`.

### 5.2 Training Commands (Consistent with Complete Pipeline Script)

```bash
# Baseline BERT (no DP, finetune encoder)
python scripts/train_bert_nli_on_A.py \
  --train_jsonl corpus_A/A_train_leaky.jsonl \
  --val_jsonl   corpus_A/A_val_leaky.jsonl  \
  --out_dir     ckpt_bert_A_finetune        \
  --finetune_encoder                        \
  --batch_size  32                          \
  --epochs      20
```

DP-BERT (multiple ε values):

```bash
for eps in 0.5 1.0 2.0 8.0; do
  python scripts/train_bert_nli_on_A.py \
    --train_jsonl corpus_A/A_train_leaky.jsonl \
    --val_jsonl   corpus_A/A_val_leaky.jsonl   \
    --out_dir     ckpt_bert_A_finetune_dp_eps${eps} \
    --finetune_encoder                         \
    --batch_size  8                            \
    --epochs      10                           \
    --dp_epsilon  ${eps}                       \
    --dp_delta    1e-5                         \
    --dp_max_grad_norm 1.0
done
```

The training script will record in `out_dir`:

* Best A-val performance;
* Complete DP configuration (e.g., ε, δ, C, batch size, number of epochs, etc.);
* Convenient for subsequent unified aggregation to `results/` and plotting.

---

## 6. Phase III – Domain Adaptation: DANN & DP-DANN

This phase adds **Domain Adversarial Training (DANN)** to BERT for A→B unlabeled domain adaptation, and adds DP-SGD (DP-DANN) under the same framework.

### 6.1 Architecture Overview

* **Encoder**: Reuse BERT encoder (loaded from `ckpt_bert_A_finetune`).
* **Task Head**: NLI classification head, uses A-domain labeled data during training.
* **Domain Discriminator**: Binary classification network that distinguishes A/B domains.
* **Gradient Reversal Layer (GRL)**: Adds sign flip to gradients to achieve domain-adversarial learning.
* **DP Version**: Uses `train_domain_adapt_A2B_DP.py` to wrap encoder/shared layers into Opacus DP-SGD, resulting in DP-DANN.

### 6.2 DANN (No DP)

```bash
python scripts/train_domain_adapt_A2B.py \
  --A_train_jsonl       corpus_A/A_train_leaky.jsonl \
  --A_val_jsonl         corpus_A/A_val_leaky.jsonl   \
  --B_train_jsonl       corpus_B/B_train.jsonl       \
  --B_val_jsonl         corpus_B/B_val.jsonl         \
  --pretrained_ckpt_dir ckpt_bert_A_finetune         \
  --out_dir             ckpt_A2B_dann                \
  --epochs              10                           \
  --lambda_dom          0.1                          \
  --lr                  1e-4
```

### 6.3 DP-DANN (DP Domain Adversarial Training)

```bash
for eps in 0.01 0.05 0.1 0.5 1.0 2.0 4.0 8.0; do
  python scripts/train_domain_adapt_A2B_DP.py \
    --A_train_jsonl       corpus_A/A_train_leaky.jsonl \
    --A_val_jsonl         corpus_A/A_val_leaky.jsonl   \
    --B_train_jsonl       corpus_B/B_train.jsonl       \
    --B_val_jsonl         corpus_B/B_val.jsonl         \
    --pretrained_ckpt_dir ckpt_bert_A_finetune         \
    --out_dir             ckpt_A2B_dp_eps${eps}        \
    --epochs              10                           \
    --dp_epsilon          ${eps}                       \
    --dp_delta            1e-5                         \
    --dp_max_grad_norm    1.0                          \
    --lambda_dom          0.1                          \
    --lr                  1e-4
done
```

`ckpt_A2B_dann` and `ckpt_A2B_dp_eps*` will be called by `eval_on_B.py` and `attack_membership_inference.py` in subsequent steps.

---

## 7. Evaluation: Utility on B

### 7.1 Classification Performance on B-test

We uniformly use the script `eval_on_B.py` to evaluate baseline / DP-BERT / DANN / DP-DANN on **B_test.clean.jsonl**.

**BERT baseline & DP-BERT:**

```bash
for ckpt in ckpt_bert_A_finetune \
            ckpt_bert_A_finetune_dp_eps0.5 \
            ckpt_bert_A_finetune_dp_eps1.0 \
            ckpt_bert_A_finetune_dp_eps2.0 \
            ckpt_bert_A_finetune_dp_eps8.0; do
  python scripts/eval_on_B.py \
    --B_test_jsonl corpus_B/B_test.clean.jsonl \
    --ckpt_dir     ${ckpt}                     \
    --model_type   baseline
done
```

**DANN & DP-DANN:**

```bash
for ckpt in ckpt_A2B_dann \
            ckpt_A2B_dp_eps0.5 \
            ckpt_A2B_dp_eps1.0 \
            ckpt_A2B_dp_eps2.0 \
            ckpt_A2B_dp_eps8.0; do
  python scripts/eval_on_B.py \
    --B_test_jsonl corpus_B/B_test.clean.jsonl \
    --ckpt_dir     ${ckpt}                     \
    --model_type   dann
done
```

Typical outputs include:

* B-test accuracy / macro-F1;
* Performance gap between A-val / B-test;
* B-domain performance curves under different ε values, enabling utility–privacy trade-off plotting.

---

## 8. Evaluation: Privacy Attacks & PHI Scan

### 8.1 Membership Inference Attack (MIA)

On domain A, we use `attack_membership_inference.py` to perform MIA on each model:

```bash
# MIA on BERT baseline & DP-BERT
for ckpt in ckpt_bert_A_finetune \
            ckpt_bert_A_finetune_dp_eps0.5 \
            ckpt_bert_A_finetune_dp_eps1.0 \
            ckpt_bert_A_finetune_dp_eps2.0 \
            ckpt_bert_A_finetune_dp_eps8.0; do
  python scripts/attack_membership_inference.py \
    --ckpt_dir   ${ckpt}                         \
    --model_type baseline                        \
    --train_jsonl corpus_A/A_train_leaky.jsonl   \
    --test_jsonl  corpus_A/A_test_leaky.jsonl
done

# MIA on DANN & DP-DANN
for ckpt in ckpt_A2B_dann \
            ckpt_A2B_dp_eps0.5 \
            ckpt_A2B_dp_eps1.0 \
            ckpt_A2B_dp_eps2.0 \
            ckpt_A2B_dp_eps8.0; do
  python scripts/attack_membership_inference.py \
    --ckpt_dir   ${ckpt}                         \
    --model_type dann                            \
    --train_jsonl corpus_A/A_train_leaky.jsonl   \
    --test_jsonl  corpus_A/A_test_leaky.jsonl
done
```

Typical script outputs include:

* Attack ROC-AUC;
* TPR / FPR under different thresholds;
* Can be directly used to plot "MIA success rate vs ε" curves.

### 8.2 PHI-like Fragment Scanning (Scan for PHI-like Content)

`scan_phi_like.py` is used to:

* Scan for possible PHI-like fragments in **B_test.clean.jsonl** (sanity check);
* Can also scan model outputs, counting canary / PHI fragment hits.

Example (usage in the pipeline script):

```bash
python scripts/scan_phi_like.py corpus_B/B_test.clean.jsonl
```

The script can output hit statistics (tp/fp/tn/fn) and precision / recall metrics, helping to demonstrate:

* The synthesized B domain contains almost no new PHI;
* Whether DP training reduces the model's tendency to memorize PHI-like content.

---

## 9. Utility–Privacy Trade-off Plots (plot_tradeoff.py)

All utility & privacy experimental results (e.g., CSV files from Eval/MIA/PHI scan) are uniformly aggregated in `results/`.
`plot_tradeoff.py` reads these files and generates ICML-style plots and tables:

```bash
python scripts/plot_tradeoff.py --summary
```

Target outputs include:

1. **Utility–Privacy Curves**

   * x-axis: ε;
   * y-axis 1: B-test Accuracy or Macro-F1;
   * y-axis 2: MIA attack AUC / canary hit rate and other privacy attack metrics;
   * Curve comparison:

     * BERT baseline (no DA) + DP-BERT;
     * BERT-DANN (no DP);
     * DP-DANN (multiple ε).

2. **A vs B Privacy Comparison Table**

   * For the same model, privacy attack success rates on A domain vs cross-domain (observed from B output);
   * Analysis of the "attack A, observe B" phenomenon and DP's suppression effect.

---
