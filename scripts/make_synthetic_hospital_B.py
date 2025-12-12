#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 A 的 JSONL 合成 Hospital B 数据：
- 术语/拼写差异（英/美式、缩写变化）
- 生命体征格式变化（单位/标点/顺序）
- 文风扰动（大小写、冗余词、科室/设备前缀、模板开头）
- class label 不变
"""

import json, random, argparse
from pathlib import Path

SEED=2025
random.seed(SEED)

REPLACE_MAP = [
    # 术语与拼写（示例，可扩展）
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
    # 低概率大小写变化
    if random.random() < 0.2:
        t = t.lower()
    if random.random() < 0.2:
        t = t.capitalize()

    # 术语替换
    for a, b in REPLACE_MAP:
        t = t.replace(a, b)

    # 生命体征格式扰动：简单正则-free 替换（示例化）
    t = t.replace("HR ", "HeartRate: ")
    t = t.replace("BP ", "BP=")
    t = t.replace("RR ", "RespRate=")
    t = t.replace("T ", "Temp=")
    t = t.replace("sat", "sats")

    # 随机加 filler 或前缀
    if random.random() < 0.5:
        t = random.choice(FILLERS) + " " + t
    if random.random() < 0.6:
        t = random.choice(PREFIXES) + t

    # 随机插入少量符号
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
