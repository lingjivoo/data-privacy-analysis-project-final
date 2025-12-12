#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制Utility-Privacy Trade-off曲线
从results/目录读取CSV文件，生成图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import json

def load_results():
    """加载所有结果CSV文件"""
    results_dir = Path("results")
    results = {}
    
    # 加载基线结果
    baseline_file = results_dir / "baseline_A_results.csv"
    if baseline_file.exists():
        results['baseline'] = pd.read_csv(baseline_file)
    
    # 加载DANN结果
    dann_file = results_dir / "dann_A2B_results.csv"
    if dann_file.exists():
        results['dann'] = pd.read_csv(dann_file)
    
    # 加载B域评估结果
    eval_file = results_dir / "eval_B_results.csv"
    if eval_file.exists():
        results['eval_B'] = pd.read_csv(eval_file)
    
    # 加载MIA攻击结果
    mia_file = results_dir / "mia_results.csv"
    if mia_file.exists():
        results['mia'] = pd.read_csv(mia_file)
    
    return results

def plot_utility_privacy_tradeoff(results, output_dir="results"):
    """绘制Utility-Privacy Trade-off曲线"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 合并数据
    if 'eval_B' in results and 'mia' in results:
        # 通过ckpt_dir合并
        eval_df = results['eval_B']
        mia_df = results['mia']
        
        # 提取epsilon值
        eval_df['epsilon'] = eval_df['dp_epsilon'].apply(lambda x: float(x) if str(x) != 'none' and x != 'none' else 0.0)
        mia_df['epsilon'] = mia_df['dp_epsilon'].apply(lambda x: float(x) if str(x) != 'none' and x != 'none' else 0.0)
        
        # 合并
        merged = pd.merge(eval_df, mia_df, on='ckpt_dir', suffixes=('_eval', '_mia'), how='inner')
        
        if len(merged) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # 左图：Utility vs Epsilon
            for model_type in merged['model_type_eval'].unique():
                subset = merged[merged['model_type_eval'] == model_type]
                ax1.plot(subset['epsilon'], subset['B_test_acc'], 'o-', label=f'{model_type} (B-test Acc)')
            
            ax1.set_xlabel('DP Epsilon (ε)')
            ax1.set_ylabel('B-test Accuracy')
            ax1.set_title('Utility vs Privacy Budget')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 右图：Privacy Attack Success vs Epsilon
            for model_type in merged['model_type_eval'].unique():
                subset = merged[merged['model_type_eval'] == model_type]
                ax2.plot(subset['epsilon'], subset['roc_auc'], 's-', label=f'{model_type} (MIA ROC-AUC)')
            
            ax2.set_xlabel('DP Epsilon (ε)')
            ax2.set_ylabel('MIA Attack Success (ROC-AUC)')
            ax2.set_title('Privacy Attack vs Privacy Budget')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'utility_privacy_tradeoff.png', dpi=300)
            print(f"Saved trade-off plot to {output_dir / 'utility_privacy_tradeoff.png'}")
    
    # 绘制单独的Utility曲线
    if 'eval_B' in results:
        eval_df = results['eval_B']
        eval_df['epsilon'] = eval_df['dp_epsilon'].apply(lambda x: float(x) if str(x) != 'none' and x != 'none' else 0.0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_type in eval_df['model_type'].unique():
            subset = eval_df[eval_df['model_type'] == model_type]
            subset = subset.sort_values('epsilon')
            ax.plot(subset['epsilon'], subset['B_test_acc'], 'o-', label=f'{model_type}')
        
        ax.set_xlabel('DP Epsilon (ε)')
        ax.set_ylabel('B-test Accuracy')
        ax.set_title('Utility on Hospital B vs Privacy Budget')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'utility_curve.png', dpi=300)
        print(f"Saved utility curve to {output_dir / 'utility_curve.png'}")
    
    # 绘制Privacy攻击曲线
    if 'mia' in results:
        mia_df = results['mia']
        mia_df['epsilon'] = mia_df['dp_epsilon'].apply(lambda x: float(x) if str(x) != 'none' and x != 'none' else 0.0)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_type in mia_df['model_type'].unique():
            subset = mia_df[mia_df['model_type'] == model_type]
            subset = subset.sort_values('epsilon')
            ax.plot(subset['epsilon'], subset['roc_auc'], 's-', label=f'{model_type}')
        
        ax.set_xlabel('DP Epsilon (ε)')
        ax.set_ylabel('MIA Attack Success (ROC-AUC)')
        ax.set_title('Privacy Attack Success vs Privacy Budget')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'privacy_attack_curve.png', dpi=300)
        print(f"Saved privacy attack curve to {output_dir / 'privacy_attack_curve.png'}")

def generate_summary_table(results, output_file="results/summary_table.csv"):
    """生成汇总表格"""
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True)
    
    summary_rows = []
    
    # 汇总基线结果
    if 'baseline' in results:
        for _, row in results['baseline'].iterrows():
            summary_rows.append({
                'experiment': row.get('experiment_id', 'unknown'),
                'model_type': 'baseline',
                'domain': 'A',
                'dp_epsilon': row.get('dp_epsilon', 'none'),
                'A_val_acc': row.get('best_val_acc', 0.0),
                'B_test_acc': 'N/A',
                'mia_roc_auc': 'N/A'
            })
    
    # 汇总评估结果
    if 'eval_B' in results:
        eval_dict = {}
        for _, row in results['eval_B'].iterrows():
            key = row['ckpt_dir']
            eval_dict[key] = {
                'B_test_acc': row.get('B_test_acc', 0.0),
                'B_test_f1': row.get('B_test_f1_macro', 0.0)
            }
        
        # 更新summary
        for row in summary_rows:
            key = row['experiment']
            if key in eval_dict:
                row['B_test_acc'] = eval_dict[key]['B_test_acc']
    
    # 汇总MIA结果
    if 'mia' in results:
        mia_dict = {}
        for _, row in results['mia'].iterrows():
            key = row['ckpt_dir']
            mia_dict[key] = row.get('roc_auc', 0.0)
        
        for row in summary_rows:
            key = row['experiment']
            if key in mia_dict:
                row['mia_roc_auc'] = mia_dict[key]
    
    # 保存汇总表
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        df.to_csv(output_file, index=False)
        print(f"Saved summary table to {output_file}")
        print("\nSummary Table:")
        print(df.to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plot utility-privacy trade-off curves")
    ap.add_argument("--output_dir", default="results", help="Output directory for plots")
    ap.add_argument("--summary", action="store_true", help="Generate summary table")
    args = ap.parse_args()
    
    results = load_results()
    
    if not results:
        print("No results found in results/ directory")
        exit(1)
    
    plot_utility_privacy_tradeoff(results, args.output_dir)
    
    if args.summary:
        generate_summary_table(results)

