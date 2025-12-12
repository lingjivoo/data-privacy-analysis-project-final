#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Synthesize Hospital B data from A's JSONL:
- Terminology/spelling differences (British/American, abbreviation changes)
- Vital signs format changes (units/punctuation/order)
- Style perturbations (case, redundant words, department/equipment prefixes, template headers)
- class label unchanged
"""

import json, random, argparse
from pathlib import Path

SEED=2025
random.seed(SEED)

REPLACE_MAP = [
    # Terminology and spelling (examples, can be extended)
    ("ed", "A&E"), ("ED", "A&E"),
    ("hemoglobin", "haemoglobin"),
    ("hematology", "haematology"),
    ("ischemia", "ischaemia"),
    ("hemorrhage", "haemorrhage"),
    ("pediatrics", "paediatrics"),
    ("O2", "SpO2"),
    ("CABG", "CABG op."),
]

PREFIXES = [
    "St. Mary's Medical Center - Ward B7 | ",
    "Queen Victoria Infirmary | ",
    "NHS Trust Hospital B | ",
    "Dept. of Respiratory Med. | ",
    "Cardio Unit (B) | ",
]

FILLERS = [
    "Per local protocol,", "As per notes,", "Reportedly,", "On review,",
    "Clinically,", "From triage,", "Observationally,"
]

def perturb_text(s):
    t = s
    # Low probability case changes
    if random.random() < 0.2:
        t = t.lower()
    if random.random() < 0.2:
        t = t.capitalize()

    # Terminology replacement
    for a, b in REPLACE_MAP:
        t = t.replace(a, b)

    # Vital signs format perturbation: simple regex-free replacement (exemplified)
    t = t.replace("HR ", "HeartRate: ")
    t = t.replace("BP ", "BP=")
    t = t.replace("RR ", "RespRate=")
    t = t.replace("T ", "Temp=")
    t = t.replace("sat", "sats")

    # Randomly add filler or prefix
    if random.random() < 0.5:
        t = random.choice(FILLERS) + " " + t
    if random.random() < 0.6:
        t = random.choice(PREFIXES) + t

    # Randomly insert a few symbols
    if random.random() < 0.3:
        t = t.replace(".", " ·")
    return t

def convert_A_to_B(in_jsonl, out_jsonl, hospital_name="Hospital B"):
    n=0
    with open(out_jsonl, "w", encoding="utf-8") as fout:
        for line in open(in_jsonl, "r", encoding="utf-8"):
            if not line.strip(): continue
            obj = json.loads(line)
            obj["sentence1"] = perturb_text(obj["sentence1"])
            obj["sentence2"] = perturb_text(obj["sentence2"])
            obj["hospital"] = hospital_name
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n+=1
    print(f"[B] {in_jsonl} -> {out_jsonl} | total={n}, hospital={hospital_name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--hospital_name", default="Hospital B")
    args = ap.parse_args()
    convert_A_to_B(args.in_jsonl, args.out_jsonl, args.hospital_name)
