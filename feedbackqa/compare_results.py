#!/usr/bin/env python3
"""
Compare Case 1 vs Case 2 Results
=================================

Compare evaluation results between baseline and experimental conditions.

Usage:
    python feedbackqa/compare_results.py \
        --case1_file outputs/case1_evaluation.json \
        --case2_file outputs/case2_evaluation.json \
        --output_file outputs/comparison_report.json
"""

import argparse
import json
from typing import Dict

import numpy as np
from scipy import stats


def load_evaluation(file_path: str) -> Dict:
    """Load evaluation results"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def extract_scores(evaluation: Dict) -> np.ndarray:
    """Extract reward scores from evaluation"""
    scores = [pred["reward_score"] for pred in evaluation["predictions"]]
    return np.array(scores)


def statistical_comparison(case1_scores: np.ndarray, case2_scores: np.ndarray) -> Dict:
    """Perform statistical comparison"""
    
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(case1_scores, case2_scores)
    
    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(case1_scores, case2_scores, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(case1_scores)**2 + np.std(case2_scores)**2) / 2)
    cohens_d = (np.mean(case2_scores) - np.mean(case1_scores)) / pooled_std
    
    # Win rate (percentage of time Case 2 > Case 1)
    # For paired comparison, we compare on same questions
    if len(case1_scores) == len(case2_scores):
        wins = np.sum(case2_scores > case1_scores)
        ties = np.sum(case2_scores == case1_scores)
        losses = np.sum(case2_scores < case1_scores)
        win_rate = wins / len(case1_scores)
    else:
        wins = ties = losses = None
        win_rate = None
    
    return {
        "t_statistic": float(t_stat),
        "t_test_pvalue": float(t_pvalue),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_pvalue": float(u_pvalue),
        "cohens_d": float(cohens_d),
        "effect_size_interpretation": interpret_cohens_d(cohens_d),
        "win_rate": float(win_rate) if win_rate is not None else None,
        "case2_wins": int(wins) if wins is not None else None,
        "ties": int(ties) if ties is not None else None,
        "case1_wins": int(losses) if losses is not None else None,
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_metrics(case1_metrics: Dict, case2_metrics: Dict) -> Dict:
    """Compare metrics between cases"""
    
    comparison = {}
    
    for metric in ["mean_reward", "std_reward", "median_reward", "min_reward", "max_reward"]:
        case1_val = case1_metrics[metric]
        case2_val = case2_metrics[metric]
        diff = case2_val - case1_val
        pct_change = (diff / abs(case1_val)) * 100 if case1_val != 0 else 0
        
        comparison[metric] = {
            "case1": case1_val,
            "case2": case2_val,
            "difference": diff,
            "percent_change": pct_change,
            "case2_better": diff > 0 if "std" not in metric else diff < 0,  # Lower std is better
        }
    
    return comparison


def print_comparison_report(
    case1_metrics: Dict,
    case2_metrics: Dict,
    metric_comparison: Dict,
    statistical_tests: Dict
):
    """Print detailed comparison report"""
    
    print("\n" + "="*80)
    print("EXPERIMENT RESULTS: Case 1 (Baseline) vs Case 2 (Feedback-Augmented)")
    print("="*80)
    
    # Overall metrics
    print("\n📊 Overall Metrics:")
    print("-"*80)
    print(f"{'Metric':<20} {'Case 1':>12} {'Case 2':>12} {'Difference':>12} {'% Change':>10}")
    print("-"*80)
    
    for metric, values in metric_comparison.items():
        symbol = "✓" if values["case2_better"] else "✗"
        print(f"{metric:<20} {values['case1']:>12.4f} {values['case2']:>12.4f} "
              f"{values['difference']:>12.4f} {values['percent_change']:>9.1f}% {symbol}")
    
    # Statistical significance
    print("\n📈 Statistical Tests:")
    print("-"*80)
    print(f"T-test:")
    print(f"  t-statistic: {statistical_tests['t_statistic']:.4f}")
    print(f"  p-value: {statistical_tests['t_test_pvalue']:.4f} {'✓ Significant' if statistical_tests['t_test_pvalue'] < 0.05 else '✗ Not significant'} (α=0.05)")
    
    print(f"\nMann-Whitney U test:")
    print(f"  U-statistic: {statistical_tests['mann_whitney_u']:.4f}")
    print(f"  p-value: {statistical_tests['mann_whitney_pvalue']:.4f} {'✓ Significant' if statistical_tests['mann_whitney_pvalue'] < 0.05 else '✗ Not significant'} (α=0.05)")
    
    print(f"\nEffect Size:")
    print(f"  Cohen's d: {statistical_tests['cohens_d']:.4f}")
    print(f"  Interpretation: {statistical_tests['effect_size_interpretation'].upper()}")
    
    if statistical_tests['win_rate'] is not None:
        print(f"\nPairwise Comparison:")
        print(f"  Case 2 wins: {statistical_tests['case2_wins']} ({statistical_tests['win_rate']*100:.1f}%)")
        print(f"  Ties: {statistical_tests['ties']}")
        print(f"  Case 1 wins: {statistical_tests['case1_wins']} ({(1-statistical_tests['win_rate'])*100:.1f}%)")
    
    # Conclusion
    print("\n🎯 Conclusion:")
    print("-"*80)
    
    mean_diff = metric_comparison['mean_reward']['difference']
    p_value = statistical_tests['t_test_pvalue']
    cohens_d = statistical_tests['cohens_d']
    
    if p_value < 0.05 and mean_diff > 0:
        print("✅ HYPOTHESIS CONFIRMED!")
        print(f"   Case 2 (Feedback-Augmented) significantly outperforms Case 1 (Baseline)")
        print(f"   Mean reward improvement: {mean_diff:.4f} ({metric_comparison['mean_reward']['percent_change']:.1f}%)")
        print(f"   Effect size: {statistical_tests['effect_size_interpretation']}")
        print(f"\n   💡 Conclusion: Training with feedback-augmented examples HELPS models learn!")
    elif p_value < 0.05 and mean_diff < 0:
        print("❌ HYPOTHESIS REJECTED!")
        print(f"   Case 1 (Baseline) significantly outperforms Case 2 (Feedback-Augmented)")
        print(f"   Mean reward difference: {mean_diff:.4f} ({metric_comparison['mean_reward']['percent_change']:.1f}%)")
        print(f"\n   💡 Conclusion: Feedback-augmented training did NOT help in this experiment.")
    else:
        print("⚠️  NO SIGNIFICANT DIFFERENCE")
        print(f"   p-value: {p_value:.4f} (> 0.05)")
        print(f"   Mean reward difference: {mean_diff:.4f} ({metric_comparison['mean_reward']['percent_change']:.1f}%)")
        print(f"\n   💡 Conclusion: Feedback-augmented training shows no significant impact.")
    
    print("="*80)


def save_comparison_report(
    case1_metrics: Dict,
    case2_metrics: Dict,
    metric_comparison: Dict,
    statistical_tests: Dict,
    output_file: str
):
    """Save comparison report to JSON"""
    
    report = {
        "case1_metrics": case1_metrics,
        "case2_metrics": case2_metrics,
        "metric_comparison": metric_comparison,
        "statistical_tests": statistical_tests,
    }
    
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Comparison report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Case 1 vs Case 2 evaluation results"
    )
    
    parser.add_argument(
        "--case1_file",
        type=str,
        required=True,
        help="Case 1 evaluation results"
    )
    parser.add_argument(
        "--case2_file",
        type=str,
        required=True,
        help="Case 2 evaluation results"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/comparison_report.json",
        help="Output comparison report"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Comparing Case 1 vs Case 2")
    print("="*80)
    
    # Load evaluations
    print(f"\nLoading Case 1: {args.case1_file}")
    case1_eval = load_evaluation(args.case1_file)
    print(f"  Loaded {len(case1_eval['predictions'])} predictions")
    
    print(f"\nLoading Case 2: {args.case2_file}")
    case2_eval = load_evaluation(args.case2_file)
    print(f"  Loaded {len(case2_eval['predictions'])} predictions")
    
    # Extract scores
    case1_scores = extract_scores(case1_eval)
    case2_scores = extract_scores(case2_eval)
    
    # Compare metrics
    metric_comparison = compare_metrics(
        case1_eval["metrics"],
        case2_eval["metrics"]
    )
    
    # Statistical tests
    statistical_tests = statistical_comparison(case1_scores, case2_scores)
    
    # Print report
    print_comparison_report(
        case1_eval["metrics"],
        case2_eval["metrics"],
        metric_comparison,
        statistical_tests
    )
    
    # Save report
    save_comparison_report(
        case1_eval["metrics"],
        case2_eval["metrics"],
        metric_comparison,
        statistical_tests,
        args.output_file
    )
    
    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80)


if __name__ == "__main__":
    main()

