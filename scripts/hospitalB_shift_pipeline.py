#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Hospital B with matched distributions across splits and stable shift vs A.

Usage:
# 1) 从 A_train 拟合全局偏移并写出 config
python scripts/hospitalB_shift_pipeline.py fit \
  --in_jsonl corpus_A/A_train_leaky.jsonl \
  --out_config corpus_B/shift_config.json

# 2) 用同一 config 逐个把 A_* 变成 B_*
python scripts/hospitalB_shift_pipeline.py transform \
  --in_jsonl corpus_A/A_train_leaky.jsonl \
  --out_jsonl corpus_B/B_train.jsonl \
  --config corpus_B/shift_config.json

python scripts/hospitalB_shift_pipeline.py transform \
  --in_jsonl corpus_A/A_val_leaky.jsonl \
  --out_jsonl corpus_B/B_val.jsonl \
  --config corpus_B/shift_config.json

python scripts/hospitalB_shift_pipeline.py transform \
  --in_jsonl corpus_A/A_test_leaky.jsonl \
  --out_jsonl corpus_B/B_test.jsonl \
  --config corpus_B/shift_config.json
"""

import json, re, random, argparse, statistics, math
from pathlib import Path
from faker import Faker

SEED = 2025
random.seed(SEED)

B_TAG = "Hospital B"

# --------- 解析正则 ----------
RE_AGE = re.compile(r'(\b)(\d{1,3})(\s*year[s]?\s+old\b)', re.IGNORECASE)
RE_T_F = re.compile(r'\bT\s*([0-9]+(?:\.[0-9]+)?)\b')            # T 98.9
RE_TEMP_EQ = re.compile(r'\bTemp\s*=\s*([0-9]+(?:\.[0-9]+)?)\b')  # Temp=37.1
RE_HR = re.compile(r'\bHR\s*[:=]?\s*(\d{2,3})\b', re.IGNORECASE)
RE_HR2= re.compile(r'\bHeartRate\s*[:=]?\s*(\d{2,3})\b', re.IGNORECASE)
RE_BP = re.compile(r'\bBP\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})\b', re.IGNORECASE)
RE_RR = re.compile(r'\bRR\s*[:=]?\s*(\d{1,2})\b', re.IGNORECASE)
RE_RR2= re.compile(r'\bRespRate\s*[:=]?\s*(\d{1,2})\b', re.IGNORECASE)
RE_SPO2_1 = re.compile(r'\bO2\s*sat\s*([0-9]+(?:\.[0-9]+)?)%\b', re.IGNORECASE)
RE_SPO2_2 = re.compile(r'\bSpO2\s*(?:sat|sats)?\s*([0-9]+(?:\.[0-9]+)?)%\b', re.IGNORECASE)


NOTE_ABBR = [
    # 常见病历短语缩写 / 改写
    (re.compile(r"\bthe patient\b", re.IGNORECASE), "pt"),
    (re.compile(r"\bpatient\b", re.IGNORECASE), "pt"),
    (re.compile(r"\bcomplains of\b", re.IGNORECASE), "c/o"),
    (re.compile(r"\breports\b", re.IGNORECASE), "c/o"),
    (re.compile(r"\bhistory of\b", re.IGNORECASE), "Hx"),
    (re.compile(r"\bhx present illness\b", re.IGNORECASE), "HPI"),
    (re.compile(r"\bshortness of breath\b", re.IGNORECASE), "SOB"),
    (re.compile(r"\bchest pain\b", re.IGNORECASE), "CP"),
    (re.compile(r"\bno chest pain\b", re.IGNORECASE), "no CP"),
    (re.compile(r"\bdiabetes mellitus\b", re.IGNORECASE), "DM"),
    (re.compile(r"\bhypertension\b", re.IGNORECASE), "HTN"),
    (re.compile(r"\bend[- ]stage renal disease\b", re.IGNORECASE), "ESRD"),
    (re.compile(r"\bmyocardial infarction\b", re.IGNORECASE), "MI"),
    (re.compile(r"\bmotor vehicle collision\b", re.IGNORECASE), "MVC"),
    (re.compile(r"\bcar accident\b", re.IGNORECASE), "MVC"),
]

NOTE_SECTIONS = ["HPI", "PMH", "O/E", "PLAN", "IMPRESSION"]

# 显式 domain token
B_DOMAIN_TAG = "[B-HOSP]"

# section 检测
SECTION_PAT = re.compile(r'\b(HPI|PMH|O/E|PLAN|IMPRESSION)\s*:\s*', re.IGNORECASE)


# 英式拼写替换与写法（transform 阶段统一做）
BRITISH_MAP = [
    ("hemoglobin", "haemoglobin"),
    ("hematology", "haematology"),
    ("ischemia", "ischaemia"),
    ("hemorrhage", "haemorrhage"),
    ("pediatrics", "paediatrics"),
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

def make_faker(locale="en_GB"):
    f = Faker(locale); Faker.seed(SEED); return f

def reorder_sections(s: str) -> str:
    """
    找到 HPI:/PMH:/O/E:/PLAN:/IMPRESSION: 这些 section block，
    把它们整体打乱顺序，改变大结构的顺序。
    """
    matches = list(SECTION_PAT.finditer(s))
    if len(matches) < 2:
        return s

    spans = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        spans.append(s[start:end].strip())

    # 随机重排 section block 的顺序
    random.shuffle(spans)
    return " ; ".join(spans)


def to_note_style(s: str) -> str:
    """
    把叙述性句子改写成更像 UK triage / ED note 的风格：
    - 使用常见缩写 (pt, c/o, Hx, SOB, CP, DM, HTN, ESRD, MVC...)
    - 用标点/连接词切成多个 clause
    - 随机重排、截断为 1–3 个 clause
    - 每个 clause 加 section: HPI:, PMH:, O/E:, PLAN:, IMPRESSION:
    """
    t = s

    # 0) 统一做一次缩写替换
    for pat, rep in NOTE_ABBR:
        t = pat.sub(rep, t)

    # 1) 用标点/连接词切分为 clause
    clauses = re.split(r"[.;]| and | but ", t)
    clauses = [c.strip() for c in clauses if c.strip()]

    if not clauses:
        return t

    # 2) 随机重排，并可缩短为前 1–3 个子句
    if len(clauses) > 1:
        random.shuffle(clauses)
        if random.random() < 0.7:  # 更高概率只保留部分 clause
            k = random.randint(1, min(3, len(clauses)))
            clauses = clauses[:k]

    # 3) 给每个 clause 随机加上一个 section label
    out_parts = []
    for c in clauses:
        if not c:
            continue
        if random.random() < 0.9:
            sec = random.choice(NOTE_SECTIONS)
            out_parts.append(f"{sec}: {c}")
        else:
            out_parts.append(c)

    return " ; ".join(out_parts)





def synth_phi(fake):
    name = fake.name()
    phone = fake.phone_number()
    email = fake.email()
    address = fake.address().replace("\n", ", ")
    mrn = f"MRN-{fake.bothify(text='??##-####')}"
    dob = fake.date_of_birth(minimum_age=0, maximum_age=95).strftime("%d/%m/%Y") # DD/MM/YYYY
    date = fake.date().replace("-", "/")
    facility = f"{B_TAG} - {fake.last_name()} Wing"
    return {
        "hospital": B_TAG, "facility": facility, "name": name, "phone": phone,
        "email": email, "address": address, "mrn": mrn, "dob": dob, "date": date
    }

def phi_to_text(phi):
    return ";".join([
        f"hospital={phi['hospital']}",
        f"facility={phi['facility']}",
        f"name={phi['name']}",
        f"phone={phi['phone']}",
        f"email={phi['email']}",
        f"address={phi['address']}",
        f"mrn={phi['mrn']}",
        f"dob={phi['dob']}",
        f"date={phi['date']}",
    ])

def _clip(x, lo, hi): return max(lo, min(hi, x))
def _mean(lst): return round(statistics.mean(lst), 3) if lst else None

# ---- 抽取 A 的简单统计（仅依赖能解析到的片段）----
def parse_stats_from_text(s, stats):
    # 年龄
    for m in RE_AGE.finditer(s):
        stats["age_A"].append(int(m.group(2)))
    # 温度（两种写法）
    for m in RE_T_F.finditer(s):
        try: stats["temp_A"].append(float(m.group(1)))
        except: pass
    for m in RE_TEMP_EQ.finditer(s):
        try: stats["temp_A"].append(float(m.group(1)))
        except: pass
    # HR
    for m in RE_HR.finditer(s):
        stats["hr_A"].append(int(m.group(1)))
    for m in RE_HR2.finditer(s):
        stats["hr_A"].append(int(m.group(1)))
    # BP
    for m in RE_BP.finditer(s):
        stats["sbp_A"].append(int(m.group(1)))
        stats["dbp_A"].append(int(m.group(2)))
    # RR
    for m in RE_RR.finditer(s):
        stats["rr_A"].append(int(m.group(1)))
    for m in RE_RR2.finditer(s):
        stats["rr_A"].append(int(m.group(1)))
    # SpO2
    for m in RE_SPO2_1.finditer(s):
        try: stats["spo2_A"].append(float(m.group(1)))
        except: pass
    for m in RE_SPO2_2.finditer(s):
        try: stats["spo2_A"].append(float(m.group(1)))
        except: pass

def fit_on_A(in_jsonl, out_config,
             # 数值 shift：比之前更大，制造更明显的病情偏移
             age_shift=+8.0, hr_shift=+8.0,
             sbp_shift=+15.0, dbp_shift=+10.0,
             rr_shift=+4.0, spo2_shift=-3.0,
             # 噪声略大一些
             temp_c_noise=0.3,
             vital_noise_std=1.0,
             # 风格扰动：更高概率 prefix / filler / note-style
             style_prefix_p=0.9, style_filler_p=0.9,
             style_case_lower_p=0.6, style_case_cap_p=0.3,
             style_punct_p=0.8, style_note_p=0.9,
             deterministic=False):
    """确定 B 的“目标偏移 + 噪声规模 + 风格概率”."""
    stats = {k: [] for k in ["age_A","temp_A","hr_A","sbp_A","dbp_A","rr_A","spo2_A"]}
    n = 0
    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            parse_stats_from_text(o["sentence1"], stats)
            parse_stats_from_text(o["sentence2"], stats)
            n += 1

    cfg = {
        "seed": SEED,
        "n_seen": n,
        "A_means": {k: _mean(v) for k, v in stats.items()},
        "shifts": {
            "age": age_shift,
            "hr": hr_shift,
            "sbp": sbp_shift,
            "dbp": dbp_shift,
            "rr": rr_shift,
            "spo2": spo2_shift
        },
        "noise": {
            "temp_c_noise": temp_c_noise,
            "vital_noise_std": vital_noise_std
        },
        "style": {
            "prefix_p": style_prefix_p,
            "filler_p": style_filler_p,
            "case_lower_p": style_case_lower_p,
            "case_cap_p": style_case_cap_p,
            "punct_p": style_punct_p,
            "note_style_p": style_note_p
        },
        "deterministic": deterministic
    }

    Path(out_config).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(out_config, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[FIT] Wrote config to {out_config}")
    print("[FIT] A_means:", cfg["A_means"])
    print("[FIT] planned shifts:", cfg["shifts"])
    print("[FIT] style:", cfg["style"])



# ------------- 变换：把 A -> B -----------------
def apply_style_and_british(s, cfg):
    # 英式拼写
    for a, b in BRITISH_MAP:
        s = re.sub(rf'\b{re.escape(a)}\b', b, s, flags=re.IGNORECASE)

    # 统一部分 vital key 写法
    s = re.sub(r'\bHR\b', 'HeartRate:', s)
    s = re.sub(r'\bBP\b', 'BP=', s)
    s = re.sub(r'\bRR\b', 'RespRate=', s)
    s = re.sub(r'\bO2\b', 'SpO2', s)

    # 清理奇怪的 "°c°c"
    s = re.sub(r'°c°c', '°C', s, flags=re.IGNORECASE)
    s = re.sub(r'°c', '°C', s, flags=re.IGNORECASE)

    st = cfg["style"]
    rnd = random.random

    # filler + prefix
    if rnd() < st.get("filler_p", 0.5):
        s = random.choice(FILLERS) + " " + s
    if rnd() < st.get("prefix_p", 0.5):
        s = random.choice(PREFIXES) + s

    # 大小写
    if rnd() < st.get("case_lower_p", st.get("case_p", 0.0)):
        s = s.lower()
    if rnd() < st.get("case_cap_p", st.get("case_p", 0.0)):
        s = s.capitalize()

    # 标点
    if rnd() < st.get("punct_p", 0.25):
        s = s.replace(".", " ·")

    # 注意：这里不再加 [B-HOSP]，在 transform_text 统一处理
    return s




def transform_text(s, cfg, collect):
    sh = cfg["shifts"]; nz = cfg["noise"]; det = cfg.get("deterministic", False)

    def n01(std):
        return 0.0 if det else random.gauss(0.0, std)

    # 局部记录 B 侧的 vitals，用来构造 ED_SUMMARY 头
    statsB = {
        "age_B": None,
        "temp_B": None,
        "hr_B": None,
        "sbp_B": None,
        "dbp_B": None,
        "rr_B": None,
        "spo2_B": None,
    }

    # 年龄
    def _age_rep(m):
        val = int(m.group(2))
        collect["age_A"].append(val)
        new = int(_clip(val + sh["age"] + n01(0.5), 0, 120))
        collect["age_B"].append(new)
        statsB["age_B"] = new
        return f"{m.group(1)}{new}{m.group(3)}"
    s = RE_AGE.sub(_age_rep, s)

    # 温度：F 或 C → 统一为 C + 小噪声
    def _temp_to_c(v):
        # 90-110 视作华氏，否则当作摄氏
        if 90 <= v <= 110:
            c = (v - 32) * 5.0/9.0
        else:
            c = v
        c = _clip(c + (0.0 if cfg["deterministic"] else random.gauss(0.1, nz["temp_c_noise"])), 34.0, 41.5)
        collect["temp_B_C"].append(c)
        statsB["temp_B"] = c
        return f"Temp={c:.1f}°C"

    def _t_rep(m):
        try:
            v = float(m.group(1))
            collect["temp_A"].append(v)
            return _temp_to_c(v)
        except:
            return m.group(0)

    s = RE_T_F.sub(_t_rep, s)
    s = RE_TEMP_EQ.sub(_t_rep, s)

    # HR
    def _hr_rep(v):
        collect["hr_A"].append(v)
        new = int(_clip(round(v + sh["hr"] + n01(nz["vital_noise_std"])), 30, 180))
        collect["hr_B"].append(new)
        statsB["hr_B"] = new
        return new
    s = RE_HR.sub(lambda m: f"HeartRate: {_hr_rep(int(m.group(1)))}", s)
    s = RE_HR2.sub(lambda m: f"HeartRate: {_hr_rep(int(m.group(1)))}", s)

    # BP
    def _bp_rep(m):
        sbp = int(m.group(1)); dbp = int(m.group(2))
        collect["sbp_A"].append(sbp); collect["dbp_A"].append(dbp)
        sbp2 = int(_clip(round(sbp + sh["sbp"] + n01(nz["vital_noise_std"])), 70, 240))
        dbp2 = int(_clip(round(dbp + sh["dbp"] + n01(nz["vital_noise_std"])), 40, 140))
        collect["sbp_B"].append(sbp2); collect["dbp_B"].append(dbp2)
        statsB["sbp_B"] = sbp2; statsB["dbp_B"] = dbp2
        return f"BP={sbp2}/{dbp2}"
    s = RE_BP.sub(_bp_rep, s)

    # RR
    def _rr_new(v):
        collect["rr_A"].append(v)
        new = int(_clip(round(v + sh["rr"] + n01(nz["vital_noise_std"])), 6, 40))
        collect["rr_B"].append(new)
        statsB["rr_B"] = new
        return new
    s = RE_RR.sub(lambda m: f"RespRate={_rr_new(int(m.group(1)))}", s)
    s = RE_RR2.sub(lambda m: f"RespRate={_rr_new(int(m.group(1)))}", s)

    # SpO2
    def _spo2_new(v):
        collect["spo2_A"].append(v)
        new = _clip(v + sh["spo2"] + n01(nz["vital_noise_std"]), 70.0, 100.0)
        collect["spo2_B"].append(new)
        statsB["spo2_B"] = new
        return new
    s = RE_SPO2_1.sub(lambda m: f"SpO2 {int(round(_spo2_new(float(m.group(1))))) }%", s)
    s = RE_SPO2_2.sub(lambda m: f"SpO2 {int(round(_spo2_new(float(m.group(1))))) }%", s)

    # 先做英式拼写 + prefix/filler/大小写/标点风格
    narrative = apply_style_and_british(s, cfg)

    # 再把 narrative 改成 triage note 风格 + section 重排
    st = cfg["style"]
    if random.random() < st.get("note_style_p", 0.95):
        narrative = to_note_style(narrative)
    if random.random() < st.get("section_reorder_p", 0.9):
        narrative = reorder_sections(narrative)

    # 构造 structured header：ED_SUMMARY
    header_parts = [B_DOMAIN_TAG, "ED_SUMMARY:"]

    ageB = statsB["age_B"]
    if ageB is not None:
        header_parts.append(f"AGE={ageB}y")

    hrB = statsB["hr_B"]
    if hrB is not None:
        header_parts.append(f"HR={hrB}")

    sbpB, dbpB = statsB["sbp_B"], statsB["dbp_B"]
    if sbpB is not None and dbpB is not None:
        header_parts.append(f"BP={sbpB}/{dbpB}")

    rrB = statsB["rr_B"]
    if rrB is not None:
        header_parts.append(f"RR={rrB}")

    tempB = statsB["temp_B"]
    if tempB is not None:
        header_parts.append(f"TEMP={tempB:.1f}°C")

    spo2B = statsB["spo2_B"]
    if spo2B is not None:
        header_parts.append(f"SpO2={int(round(spo2B))}%")

    header = " ".join(header_parts)

    # 最终串：structured header + narrative
    return f"{header} || {narrative}"


def summarize(collect, title):
    def mean(x): return round(statistics.mean(x), 2) if x else None
    print(f"\n--- {title} ---")
    for kA, kB, desc in [
        ("age_A","age_B","Age (year)"),
        ("hr_A","hr_B","HeartRate (bpm)"),
        ("sbp_A","sbp_B","Systolic BP (mmHg)"),
        ("dbp_A","dbp_B","Diastolic BP (mmHg)"),
        ("rr_A","rr_B","RespRate (/min)"),
        ("spo2_A","spo2_B","SpO2 (%)"),
        ("temp_A","temp_B_C","Temp (A raw ~ F/C, B °C)")
    ]:
        print(f"{desc:24s}: A mean={mean(collect.get(kA,[]))} | B mean={mean(collect.get(kB,[]))}")

def transform_file(in_jsonl, out_jsonl, config_path):
    cfg = json.load(open(config_path, "r", encoding="utf-8"))
    random.seed(cfg["seed"])  # 可复现
    fake = make_faker("en_GB")

    collect = {k:[] for k in [
        "age_A","age_B","hr_A","hr_B","sbp_A","sbp_B","dbp_A","dbp_B",
        "rr_A","rr_B","spo2_A","spo2_B","temp_A","temp_B_C"
    ]}
    n=0
    with open(in_jsonl, "r", encoding="utf-8") as f, open(out_jsonl, "w", encoding="utf-8") as g:
        for line in f:
            if not line.strip(): continue
            o = json.loads(line)

            s1 = transform_text(o["sentence1"], cfg, collect)
            s2 = transform_text(o["sentence2"], cfg, collect)

            # 重新生成 B 的合成 PHI
            phi = synth_phi(fake)
            out = {
                "sentence1": s1,
                "sentence2": s2,
                "gold_label": o["gold_label"],
                "phi_text": phi_to_text(phi),
                "leak_flag": 1,            # B 侧也带合成 PHI
                "hospital": B_TAG
            }
            g.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
    print(f"[TRANSFORM] {in_jsonl} -> {out_jsonl} | {n} rows")
    summarize(collect, f"Distribution Summary for {Path(out_jsonl).name}")

# ------------- CLI ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fit","transform"])
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_config")
    ap.add_argument("--out_jsonl")
    ap.add_argument("--config")
    ap.add_argument("--deterministic", action="store_true", help="fit 时设置为确定性(不加样本噪声)")
    args = ap.parse_args()

    if args.mode == "fit":
        assert args.out_config, "--out_config 必填"
        fit_on_A(args.in_jsonl, args.out_config, deterministic=args.deterministic)
    else:
        assert args.config and args.out_jsonl, "--config 与 --out_jsonl 必填"
        transform_file(args.in_jsonl, args.out_jsonl, args.config)
