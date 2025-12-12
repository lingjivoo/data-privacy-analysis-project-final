#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扫描文本中的PHI-like片段（用于隐私攻击评估）
支持从JSONL文件或标准输入读取，输出TP/FP/TN/FN统计
"""

import json, argparse, re, sys
from pathlib import Path
from collections import Counter

# PHI模式：姓名、电话、邮箱、MRN、日期等
PHI_PATTERNS = [
    (re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'), 'name'),  # 简单姓名模式
    (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), 'phone'),  # 电话
    (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'), 'email'),  # 邮箱
    (re.compile(r'\bMRN[-]?[A-Z0-9-]+\b', re.IGNORECASE), 'mrn'),  # MRN
    (re.compile(r'\b\d{4}[-/]\d{2}[-/]\d{2}\b'), 'date'),  # 日期 YYYY-MM-DD
    (re.compile(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'), 'date_alt'),  # 日期 MM/DD/YYYY
    (re.compile(r'\b\d{1,3}\s+[A-Z][a-z]+\s+Street\b', re.IGNORECASE), 'address'),  # 地址
]

def extract_phi_spans(text):
    """从文本中提取所有PHI-like片段"""
    spans = []
    for pattern, phi_type in PHI_PATTERNS:
        for match in pattern.finditer(text):
            spans.append({
                'text': match.group(0),
                'type': phi_type,
                'start': match.start(),
                'end': match.end()
            })
    return spans

def check_phi_in_text(text, phi_text=""):
    """检查文本中是否包含PHI信息"""
    if not phi_text:
        return False, []
    
    # 提取文本中的PHI片段
    spans = extract_phi_spans(text)
    
    # 检查是否与phi_text有重叠
    phi_words = set(phi_text.lower().split())
    found_phi = False
    matched_spans = []
    
    for span in spans:
        span_words = set(span['text'].lower().split())
        if span_words & phi_words:  # 有交集
            found_phi = True
            matched_spans.append(span)
    
    return found_phi, matched_spans

def scan_jsonl_file(in_path, output_stats=True):
    """扫描JSONL文件中的PHI-like片段"""
    tp = fp = tn = fn = 0
    all_spans = Counter()
    
    with open(in_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                text = (obj.get('sentence1', '') + ' ' + obj.get('sentence2', '')).strip()
                phi_text = obj.get('phi_text', '')
                leak_flag = obj.get('leak_flag', 0)
                
                has_phi, spans = check_phi_in_text(text, phi_text)
                
                # 统计
                for span in spans:
                    all_spans[span['type']] += 1
                
                # TP: leak_flag=1 且检测到PHI
                # FP: leak_flag=0 但检测到PHI
                # TN: leak_flag=0 且未检测到PHI
                # FN: leak_flag=1 但未检测到PHI
                if leak_flag == 1:
                    if has_phi:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if has_phi:
                        fp += 1
                    else:
                        tn += 1
                        
            except json.JSONDecodeError:
                continue
    
    if output_stats:
        total = tp + fp + tn + fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"\n[PHI Scan Results] {in_path}")
        print(f"  Total samples: {total}")
        print(f"  TP (True Positive): {tp}")
        print(f"  FP (False Positive): {fp}")
        print(f"  TN (True Negative): {tn}")
        print(f"  FN (False Negative): {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1: {f1:.4f}")
        print(f"\n  PHI type distribution:")
        for phi_type, count in all_spans.most_common():
            print(f"    {phi_type}: {count}")
    
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'phi_spans': dict(all_spans)
    }

def scan_stdin():
    """从标准输入读取并扫描"""
    tp = fp = tn = fn = 0
    all_spans = Counter()
    
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            text = (obj.get('sentence1', '') + ' ' + obj.get('sentence2', '')).strip()
            phi_text = obj.get('phi_text', '')
            leak_flag = obj.get('leak_flag', 0)
            
            has_phi, spans = check_phi_in_text(text, phi_text)
            
            for span in spans:
                all_spans[span['type']] += 1
            
            if leak_flag == 1:
                if has_phi:
                    tp += 1
                else:
                    fn += 1
            else:
                if has_phi:
                    fp += 1
                else:
                    tn += 1
        except json.JSONDecodeError:
            continue
    
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"\n[PHI Scan Results] (stdin)")
    print(f"  Total samples: {total}")
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"  PHI type distribution: {dict(all_spans)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scan for PHI-like patterns in JSONL files")
    ap.add_argument("input_file", nargs="?", help="Input JSONL file (if not provided, read from stdin)")
    ap.add_argument("--output", help="Output JSON file with statistics")
    args = ap.parse_args()
    
    if args.input_file:
        stats = scan_jsonl_file(args.input_file)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(stats, f, indent=2)
    else:
        scan_stdin()

