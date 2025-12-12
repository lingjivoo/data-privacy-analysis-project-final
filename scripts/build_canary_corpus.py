#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From MedNLI JSONL -> LM corpus with labels + synthetic canaries.

Each sample written as one line of training text (causal LM can learn directly):
NOTE: <s1> [SEP] <s2> \nLabel: <gold_label> \nContact info: <PHI>key=val;... </PHI>

Export:
  <out_dir>/train.txt, val.txt, test.txt                 # Plain text for LM training/validation/testing
  <out_dir>/train_canaries.csv etc. (contains gold_label, prompt, phi_text)
  <out_dir>/cls_train.jsonl etc. (classification supervised pairs, prompt->gold_label)
"""

import json, argparse, random, csv
from pathlib import Path
from faker import Faker

SEED = 2025
random.seed(SEED)
LABELS = {"entailment","contradiction","neutral"}

def synth_phi(fake, hospital="Hospital A"):
    name=fake.name(); phone=fake.phone_number(); email=fake.email()
    address=fake.address().replace("\n", ", ")
    mrn=f"MRN-{fake.bothify(text='??##-####')}"
    dob=fake.date_of_birth(minimum_age=0, maximum_age=95).strftime("%Y-%m-%d")
    date=fake.date().replace("-", "/"); wing=f"{hospital} - {fake.last_name()} Wing"
    return {"hospital":hospital, "facility":wing, "name":name, "phone":phone,
            "email":email, "address":address, "mrn":mrn, "dob":dob, "date":date}

def phi_block(phi: dict) -> str:
    return "<PHI>" + ";".join([f"{k}={v}" for k,v in phi.items()]) + "</PHI>"

def one_line(s1, s2, y, phi, trigger="Contact info: "):
    body = f"NOTE: {s1.strip()} [SEP] {s2.strip()}"
    return f"{body}\nLabel: {y}\n{trigger}{phi_block(phi)}"

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)
            assert o["gold_label"] in LABELS, f"bad label: {o['gold_label']}"
            yield o

def process_split(in_jsonl, out_txt, out_canary_csv, out_cls_jsonl,
                  fake, hospital, trigger):
    n = 0
    with open(out_txt, "w", encoding="utf-8") as ftxt, \
         open(out_canary_csv, "w", newline="", encoding="utf-8") as fcsv, \
         open(out_cls_jsonl, "w", encoding="utf-8") as fcls:
        wr = csv.writer(fcsv)
        wr.writerow(["canary_id","gold_label","prompt","phi_text"])
        can_id = 0
        for o in load_jsonl(in_jsonl):
            y = o["gold_label"]; phi = synth_phi(fake, hospital)
            line = one_line(o["sentence1"], o["sentence2"], y, phi, trigger)
            ftxt.write(line + "\n")

            # prompt (prefix fed to model during attack/classification, does not contain <PHI>)
            prompt = line.split("<PHI>")[0]
            phi_text = line[len(prompt):].strip()
            wr.writerow([can_id, y, prompt, phi_text]); can_id += 1

            # Classification supervised pairs: use prompt up to "Label: " line as input
            cls_input = prompt.split("\nLabel:")[0] + "\nLabel: "
            fcls.write(json.dumps({"x": cls_input, "y": y}, ensure_ascii=False) + "\n")
            n += 1
    return n

def main(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fake = Faker(args.locale); Faker.seed(SEED)

    def run(split, in_path):
        if not in_path: return
        n = process_split(
            in_path,
            out / f"{split}.txt",
            out / f"{split}_canaries.csv",
            out / f"cls_{split}.jsonl",
            fake, args.hospital, args.trigger
        )
        print(f"[OK] {split}: {n}")

    run("train", args.train_jsonl)
    run("val",   args.val_jsonl)
    run("test",  args.test_jsonl)
    print(f"[DONE] out_dir={out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl",   required=True)
    ap.add_argument("--test_jsonl",  default="")
    ap.add_argument("--out_dir",     default="corpus_A")
    ap.add_argument("--trigger",     default="Contact info: ")
    ap.add_argument("--hospital",    default="Hospital A")
    ap.add_argument("--locale",      default="en_US")
    args = ap.parse_args()
    main(args)
