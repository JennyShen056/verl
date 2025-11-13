"""
Compare evaluation results between Case 1 and Case 2 for HelpSteer3.
Performs statistical tests and generates comparison report.
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
from scipy import stats


def load_evaluation(file_path: str) -> Dict:
    """Load evaluation results from JSON file"""
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def paired_t_test(rewards1: List[float], rewards2: List[float]) -> Dict:
    """Perform paired t-test"""
    t_stat, p_value = stats.ttest_rel(rewards1, rewards2)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),  # Convert to native Python bool
        "significance_level": "p < 0.05" if p_value < 0.05 else "not significant"
    }


def wilcoxon_test(rewards1: List[float], rewards2: List[float]) -> Dict:
    """Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test)"""
    stat, p_value = stats.wilcoxon(rewards1, rewards2)
    
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),  # Convert to native Python bool
        "significance_level": "p < 0.05" if p_value < 0.05 else "not significant"
    }


def cohen_d(rewards1: List[float], rewards2: List[float]) -> float:
    """Calculate Cohen's d effect size"""
    diff = np.array(rewards1) - np.array(rewards2)
    return float(np.mean(diff) / np.std(diff, ddof=1))


def compare_evaluations(case1_data: Dict, case2_data: Dict) -> Dict:
    """Compare two evaluation results"""
    
    # Extract statistics
    case1_stats = case1_data["statistics"]
    case2_stats = case2_data["statistics"]
    
    # Extract predictions
    case1_preds = case1_data["predictions"]
    case2_preds = case2_data["predictions"]
    
    # Ensure same number of predictions
    assert len(case1_preds) == len(case2_preds), "Predictions must be from same test set"
    
    # Extract reward scores
    case1_rewards = [p["reward_score"] for p in case1_preds]
    case2_rewards = [p["reward_score"] for p in case2_preds]
    
    # Compute difference
    reward_diff = np.array(case2_rewards) - np.array(case1_rewards)
    
    # Statistical tests
    t_test_results = paired_t_test(case1_rewards, case2_rewards)
    wilcoxon_results = wilcoxon_test(case1_rewards, case2_rewards)
    effect_size = cohen_d(case2_rewards, case1_rewards)
    
    # Build comparison report
    comparison = {
        "summary": {
            "case1_mean_reward": case1_stats["mean_reward"],
            "case2_mean_reward": case2_stats["mean_reward"],
            "absolute_improvement": case2_stats["mean_reward"] - case1_stats["mean_reward"],
            "relative_improvement_pct": (
                (case2_stats["mean_reward"] - case1_stats["mean_reward"]) / 
                abs(case1_stats["mean_reward"]) * 100
            ),
            "num_examples": len(case1_preds),
        },
        "detailed_statistics": {
            "case1": {
                "mean": case1_stats["mean_reward"],
                "median": case1_stats["median_reward"],
                "min": case1_stats["min_reward"],
                "max": case1_stats["max_reward"],
                "std": float(np.std(case1_rewards)),
            },
            "case2": {
                "mean": case2_stats["mean_reward"],
                "median": case2_stats["median_reward"],
                "min": case2_stats["min_reward"],
                "max": case2_stats["max_reward"],
                "std": float(np.std(case2_rewards)),
            },
            "difference": {
                "mean_diff": float(np.mean(reward_diff)),
                "median_diff": float(np.median(reward_diff)),
                "std_diff": float(np.std(reward_diff)),
                "examples_improved": int(np.sum(reward_diff > 0)),
                "examples_worse": int(np.sum(reward_diff < 0)),
                "examples_same": int(np.sum(reward_diff == 0)),
            }
        },
        "statistical_tests": {
            "paired_t_test": t_test_results,
            "wilcoxon_signed_rank_test": wilcoxon_results,
            "effect_size_cohens_d": float(effect_size),
            "effect_size_interpretation": (
                "large" if abs(effect_size) >= 0.8 else
                "medium" if abs(effect_size) >= 0.5 else
                "small" if abs(effect_size) >= 0.2 else
                "negligible"
            )
        },
        "domain_comparison": {}
    }
    
    # Domain-wise comparison
    for domain in case1_stats["by_domain"]:
        if domain in case2_stats["by_domain"]:
            comparison["domain_comparison"][domain] = {
                "case1_mean": case1_stats["by_domain"][domain]["mean_reward"],
                "case2_mean": case2_stats["by_domain"][domain]["mean_reward"],
                "improvement": (
                    case2_stats["by_domain"][domain]["mean_reward"] -
                    case1_stats["by_domain"][domain]["mean_reward"]
                ),
                "count": case1_stats["by_domain"][domain]["count"]
            }
    
    return comparison


def print_comparison_report(comparison: Dict):
    """Print formatted comparison report"""
    
    print("\n" + "="*80)
    print("COMPARISON REPORT: Case 1 (Question Only) vs Case 2 (With Feedback)")
    print("="*80)
    
    summary = comparison["summary"]
    print(f"\nNumber of test examples: {summary['num_examples']}")
    print(f"\nCase 1 (Baseline) mean reward: {summary['case1_mean_reward']:.4f}")
    print(f"Case 2 (With Feedback) mean reward: {summary['case2_mean_reward']:.4f}")
    print(f"\nAbsolute improvement: {summary['absolute_improvement']:+.4f}")
    print(f"Relative improvement: {summary['relative_improvement_pct']:+.2f}%")
    
    print("\n" + "-"*80)
    print("STATISTICAL SIGNIFICANCE")
    print("-"*80)
    
    t_test = comparison["statistical_tests"]["paired_t_test"]
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_test['t_statistic']:.4f}")
    print(f"  p-value: {t_test['p_value']:.6f}")
    print(f"  Result: {'✓ SIGNIFICANT' if t_test['significant'] else '✗ NOT SIGNIFICANT'} ({t_test['significance_level']})")
    
    wilcoxon = comparison["statistical_tests"]["wilcoxon_signed_rank_test"]
    print(f"\nWilcoxon signed-rank test:")
    print(f"  statistic: {wilcoxon['statistic']:.4f}")
    print(f"  p-value: {wilcoxon['p_value']:.6f}")
    print(f"  Result: {'✓ SIGNIFICANT' if wilcoxon['significant'] else '✗ NOT SIGNIFICANT'} ({wilcoxon['significance_level']})")
    
    effect_size = comparison["statistical_tests"]["effect_size_cohens_d"]
    interpretation = comparison["statistical_tests"]["effect_size_interpretation"]
    print(f"\nEffect size (Cohen's d): {effect_size:.4f} ({interpretation})")
    
    print("\n" + "-"*80)
    print("DETAILED STATISTICS")
    print("-"*80)
    
    diff = comparison["detailed_statistics"]["difference"]
    print(f"\nExamples improved (Case 2 > Case 1): {diff['examples_improved']}")
    print(f"Examples worse (Case 2 < Case 1): {diff['examples_worse']}")
    print(f"Examples same: {diff['examples_same']}")
    
    print("\n" + "-"*80)
    print("DOMAIN-WISE COMPARISON")
    print("-"*80)
    
    for domain, domain_stats in comparison["domain_comparison"].items():
        print(f"\n{domain} (n={domain_stats['count']}):")
        print(f"  Case 1: {domain_stats['case1_mean']:.4f}")
        print(f"  Case 2: {domain_stats['case2_mean']:.4f}")
        print(f"  Improvement: {domain_stats['improvement']:+.4f}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if t_test['significant']:
        direction = "better" if summary['absolute_improvement'] > 0 else "worse"
        print(f"\n✓ Case 2 (With Feedback) performs SIGNIFICANTLY {direction} than Case 1 (Question Only)")
        print(f"  The improvement of {summary['absolute_improvement']:+.4f} is statistically significant (p < 0.05)")
        print(f"  Effect size is {interpretation} (Cohen's d = {effect_size:.4f})")
    else:
        print(f"\n✗ No significant difference found between Case 1 and Case 2")
        print(f"  The difference of {summary['absolute_improvement']:+.4f} is not statistically significant")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare Case 1 and Case 2 evaluation results")
    parser.add_argument("--case1_file", type=str, required=True, help="Case 1 evaluation JSON")
    parser.add_argument("--case2_file", type=str, required=True, help="Case 2 evaluation JSON")
    parser.add_argument("--output_file", type=str, required=True, help="Output comparison JSON")
    
    args = parser.parse_args()
    
    print("="*80)
    print("HelpSteer3 Results Comparison")
    print("="*80)
    
    # Load evaluation results
    case1_data = load_evaluation(args.case1_file)
    case2_data = load_evaluation(args.case2_file)
    
    # Compare
    print("\nComparing results...")
    comparison = compare_evaluations(case1_data, case2_data)
    
    # Print report
    print_comparison_report(comparison)
    
    # Save comparison
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✓ Saved comparison report to {args.output_file}")


if __name__ == "__main__":
    main()

