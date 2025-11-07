#!/usr/bin/env python3
"""
Qualitative Analysis Sample Selection
======================================

Analyzes Case 1 vs Case 2 evaluation results and selects representative
examples for qualitative analysis.

Usage:
    python feedbackqa/analyze_qualitative_samples.py \
        --case1_file outputs/case1_evaluation.json \
        --case2_file outputs/case2_evaluation.json \
        --output_file outputs/qualitative_samples.json
"""

import argparse
import json
import statistics
from typing import Dict, List, Tuple


def load_evaluation(file_path: str) -> Dict:
    """Load evaluation results"""
    with open(file_path, 'r') as f:
        return json.load(f)


def compute_reward_differences(case1_data: Dict, case2_data: Dict) -> List[Dict]:
    """Compute reward differences between Case 1 and Case 2"""
    
    results = []
    
    for i, (pred1, pred2) in enumerate(zip(case1_data['predictions'], case2_data['predictions'])):
        # Verify same question
        assert pred1['question'] == pred2['question'], f"Question mismatch at index {i}"
        
        reward_diff = pred2['reward_score'] - pred1['reward_score']
        
        results.append({
            'index': i,
            'question': pred1['question'],
            'ground_truth': pred1['ground_truth'],
            'feedback': pred1['feedback'],
            'rating': pred1['rating'],
            'case1_answer': pred1['generated_answer'],
            'case1_reward': pred1['reward_score'],
            'case2_answer': pred2['generated_answer'],
            'case2_reward': pred2['reward_score'],
            'reward_diff': reward_diff,
            'abs_reward_diff': abs(reward_diff),
        })
    
    return results


def categorize_examples(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize examples by reward difference patterns"""
    
    categories = {
        'case2_much_better': [],      # Case 2 significantly better (diff > 0.2)
        'case2_better': [],            # Case 2 better (0.05 < diff <= 0.2)
        'similar': [],                 # Similar performance (-0.05 <= diff <= 0.05)
        'case1_better': [],            # Case 1 better (-0.2 <= diff < -0.05)
        'case1_much_better': [],       # Case 1 significantly better (diff < -0.2)
    }
    
    for result in results:
        diff = result['reward_diff']
        
        if diff > 0.2:
            categories['case2_much_better'].append(result)
        elif diff > 0.05:
            categories['case2_better'].append(result)
        elif diff >= -0.05:
            categories['similar'].append(result)
        elif diff >= -0.2:
            categories['case1_better'].append(result)
        else:
            categories['case1_much_better'].append(result)
    
    return categories


def select_representative_samples(categories: Dict[str, List[Dict]], n_per_category: int = 3) -> Dict[str, List[Dict]]:
    """Select representative samples from each category"""
    
    representative = {}
    
    for category, examples in categories.items():
        if not examples:
            representative[category] = []
            continue
        
        # Sort by absolute reward difference (descending)
        sorted_examples = sorted(examples, key=lambda x: x['abs_reward_diff'], reverse=True)
        
        # Take top N
        representative[category] = sorted_examples[:n_per_category]
    
    return representative


def select_by_rating(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Select examples by ground truth rating"""
    
    by_rating = {
        'excellent': [],
        'good': [],
        'bad': [],
    }
    
    for result in results:
        rating = result['rating'].lower()
        if 'excellent' in rating:
            by_rating['excellent'].append(result)
        elif 'good' in rating:
            by_rating['good'].append(result)
        elif 'bad' in rating:
            by_rating['bad'].append(result)
    
    # Select top 3 by absolute reward difference from each rating category
    for rating_cat in by_rating:
        sorted_examples = sorted(by_rating[rating_cat], key=lambda x: x['abs_reward_diff'], reverse=True)
        by_rating[rating_cat] = sorted_examples[:3]
    
    return by_rating


def print_summary_statistics(case1_data: Dict, case2_data: Dict, results: List[Dict], categories: Dict):
    """Print summary statistics"""
    
    print("\n" + "="*80)
    print("QUALITATIVE ANALYSIS SUMMARY")
    print("="*80)
    
    # Overall metrics
    print("\n📊 Overall Metrics:")
    print("-"*80)
    print(f"{'Metric':<25} {'Case 1':>15} {'Case 2':>15} {'Difference':>15}")
    print("-"*80)
    
    metrics = ['mean_reward', 'std_reward', 'median_reward', 'min_reward', 'max_reward']
    for metric in metrics:
        c1_val = case1_data['metrics'][metric]
        c2_val = case2_data['metrics'][metric]
        diff = c2_val - c1_val
        symbol = "✓" if diff > 0 else "✗" if diff < 0 else "="
        print(f"{metric:<25} {c1_val:>15.4f} {c2_val:>15.4f} {diff:>14.4f} {symbol}")
    
    # Distribution of differences
    print("\n📈 Distribution of Reward Differences:")
    print("-"*80)
    reward_diffs = [r['reward_diff'] for r in results]
    print(f"Mean difference (Case 2 - Case 1): {statistics.mean(reward_diffs):.4f}")
    print(f"Median difference: {statistics.median(reward_diffs):.4f}")
    print(f"Std deviation: {statistics.stdev(reward_diffs):.4f}")
    print(f"Min difference: {min(reward_diffs):.4f}")
    print(f"Max difference: {max(reward_diffs):.4f}")
    
    # Category breakdown
    print("\n📋 Category Breakdown:")
    print("-"*80)
    print(f"{'Category':<30} {'Count':>10} {'Percentage':>15}")
    print("-"*80)
    total = len(results)
    for category, examples in categories.items():
        count = len(examples)
        pct = (count / total) * 100
        print(f"{category:<30} {count:>10} {pct:>14.1f}%")
    
    # Win/Loss/Tie summary
    print("\n🏆 Win/Loss/Tie Summary:")
    print("-"*80)
    case2_wins = len(categories['case2_much_better']) + len(categories['case2_better'])
    ties = len(categories['similar'])
    case1_wins = len(categories['case1_much_better']) + len(categories['case1_better'])
    
    print(f"Case 2 Wins: {case2_wins} ({case2_wins/total*100:.1f}%)")
    print(f"Ties: {ties} ({ties/total*100:.1f}%)")
    print(f"Case 1 Wins: {case1_wins} ({case1_wins/total*100:.1f}%)")
    
    # By rating
    print("\n⭐ Performance by Ground Truth Rating:")
    print("-"*80)
    ratings = ['Excellent', 'Good', 'Bad']
    for rating in ratings:
        rating_examples = [r for r in results if rating.lower() in r['rating'].lower()]
        if rating_examples:
            avg_c1 = statistics.mean([r['case1_reward'] for r in rating_examples])
            avg_c2 = statistics.mean([r['case2_reward'] for r in rating_examples])
            count = len(rating_examples)
            print(f"{rating} ({count} examples):")
            print(f"  Case 1 avg: {avg_c1:.4f}")
            print(f"  Case 2 avg: {avg_c2:.4f}")
            print(f"  Difference: {avg_c2 - avg_c1:.4f}")
    
    print("="*80)


def print_qualitative_samples(representative: Dict[str, List[Dict]]):
    """Print selected qualitative samples"""
    
    print("\n" + "="*80)
    print("REPRESENTATIVE SAMPLES FOR QUALITATIVE ANALYSIS")
    print("="*80)
    
    for category, examples in representative.items():
        if not examples:
            continue
        
        print(f"\n\n{'='*80}")
        print(f"CATEGORY: {category.upper().replace('_', ' ')}")
        print(f"{'='*80}")
        
        for i, ex in enumerate(examples, 1):
            print(f"\n{'-'*80}")
            print(f"Example {i} (Index {ex['index']})")
            print(f"{'-'*80}")
            print(f"\n📝 Question:")
            print(f"{ex['question']}")
            
            print(f"\n⭐ Ground Truth Rating: {ex['rating']}")
            print(f"💬 Feedback: {ex['feedback']}")
            
            print(f"\n📊 Reward Scores:")
            print(f"  Case 1: {ex['case1_reward']:.4f}")
            print(f"  Case 2: {ex['case2_reward']:.4f}")
            print(f"  Difference: {ex['reward_diff']:.4f} ({'Case 2 better' if ex['reward_diff'] > 0 else 'Case 1 better'})")
            
            print(f"\n🔵 Case 1 Answer (Question Only):")
            print(f"{ex['case1_answer'][:500]}..." if len(ex['case1_answer']) > 500 else ex['case1_answer'])
            
            print(f"\n🟢 Case 2 Answer (With Feedback):")
            print(f"{ex['case2_answer'][:500]}..." if len(ex['case2_answer']) > 500 else ex['case2_answer'])
            
            print(f"\n✅ Ground Truth:")
            print(f"{ex['ground_truth'][:300]}..." if len(ex['ground_truth']) > 300 else ex['ground_truth'])


def save_samples(representative: Dict, by_rating: Dict, output_file: str):
    """Save selected samples to JSON"""
    
    output = {
        'by_improvement_category': representative,
        'by_rating': by_rating,
        'metadata': {
            'total_samples_selected': sum(len(examples) for examples in representative.values()),
            'categories': list(representative.keys()),
            'ratings': list(by_rating.keys()),
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved qualitative samples to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and select representative samples for qualitative analysis"
    )
    
    parser.add_argument(
        "--case1_file",
        type=str,
        default="outputs/case1_evaluation.json",
        help="Case 1 evaluation file"
    )
    parser.add_argument(
        "--case2_file",
        type=str,
        default="outputs/case2_evaluation.json",
        help="Case 2 evaluation file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="outputs/qualitative_samples.json",
        help="Output file for selected samples"
    )
    parser.add_argument(
        "--n_per_category",
        type=int,
        default=3,
        help="Number of samples per category"
    )
    parser.add_argument(
        "--show_samples",
        action="store_true",
        help="Print sample details to console"
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Qualitative Analysis: Case 1 vs Case 2")
    print("="*80)
    
    # Load data
    print(f"\nLoading Case 1: {args.case1_file}")
    case1_data = load_evaluation(args.case1_file)
    print(f"  {len(case1_data['predictions'])} predictions")
    
    print(f"\nLoading Case 2: {args.case2_file}")
    case2_data = load_evaluation(args.case2_file)
    print(f"  {len(case2_data['predictions'])} predictions")
    
    # Compute differences
    print("\nComputing reward differences...")
    results = compute_reward_differences(case1_data, case2_data)
    
    # Categorize
    print("Categorizing examples...")
    categories = categorize_examples(results)
    
    # Select representative samples
    print(f"Selecting representative samples ({args.n_per_category} per category)...")
    representative = select_representative_samples(categories, args.n_per_category)
    
    # Select by rating
    print("Selecting samples by rating...")
    by_rating = select_by_rating(results)
    
    # Print statistics
    print_summary_statistics(case1_data, case2_data, results, categories)
    
    # Print samples if requested
    if args.show_samples:
        print_qualitative_samples(representative)
    
    # Save samples
    save_samples(representative, by_rating, args.output_file)
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"  1. Review {args.output_file} for selected samples")
    print(f"  2. Conduct qualitative analysis on representative examples")
    print(f"  3. Look for patterns in Case 2 improvements")
    print(f"  4. Identify where feedback-augmented training helps most")


if __name__ == "__main__":
    main()

