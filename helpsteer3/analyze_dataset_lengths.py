"""
Analyze prompt and response lengths for HelpSteer3 dataset.
Computes statistics for both Case 1 and Case 2 using Llama-3.1-8B-Instruct tokenizer.
"""

import argparse
import random
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def format_conversation(context: List[Dict]) -> str:
    """Convert conversation context list to a formatted string"""
    formatted = []
    for turn in context:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            formatted.append(f"User: {content}")
        elif role == "assistant":
            formatted.append(f"Assistant: {content}")
    return "\n\n".join(formatted)


def format_feedback_list(feedback_list: List[str]) -> str:
    """Convert feedback list to a formatted string"""
    if not feedback_list:
        return "No specific feedback provided."
    return "\n".join([f"- {fb}" for fb in feedback_list])


def create_case1_prompt(context: List[Dict]) -> List[Dict]:
    """Case 1: Question only - just the conversation context"""
    return context.copy()


def create_case2_prompt(context: List[Dict], chosen_response: str, chosen_feedback: List[str]) -> List[Dict]:
    """Case 2: With previous response and feedback"""
    formatted_context = format_conversation(context)
    formatted_feedback = format_feedback_list(chosen_feedback)
    
    # Single-turn prompt with previous response to SAME context
    prompt_content = (
        f"Here is a previous response to this conversation with feedback:\n\n"
        f"Conversation:\n{formatted_context}\n\n"
        f"Previous Response: {chosen_response}\n\n"
        f"Feedback: {formatted_feedback}\n\n"
        f"Now, please respond to the same conversation directly:\n\n"
        f"Conversation:\n{formatted_context}"
    )
    
    return [{"role": "user", "content": prompt_content}]


def compute_statistics(values: List[int], name: str) -> Dict:
    """Compute statistics for a list of values"""
    values = np.array(values)
    return {
        "name": name,
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": int(np.min(values)),
        "max": int(np.max(values)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def print_statistics(stats: Dict):
    """Print statistics in a formatted table"""
    print(f"\n{stats['name']}")
    print("-" * 80)
    print(f"  Count:      {stats['count']:,}")
    print(f"  Mean:       {stats['mean']:.1f}")
    print(f"  Median:     {stats['median']:.1f}")
    print(f"  Std Dev:    {stats['std']:.1f}")
    print(f"  Min:        {stats['min']:,}")
    print(f"  Max:        {stats['max']:,}")
    print(f"\n  Percentiles:")
    print(f"    25th:     {stats['p25']:.1f}")
    print(f"    50th:     {stats['p50']:.1f}")
    print(f"    75th:     {stats['p75']:.1f}")
    print(f"    90th:     {stats['p90']:.1f}")
    print(f"    95th:     {stats['p95']:.1f}")
    print(f"    99th:     {stats['p99']:.1f}")


def analyze_split(
    split: str,
    tokenizer,
    max_samples: int = None,
    seed: int = 42
) -> Dict:
    """Analyze a specific split of the dataset"""
    
    print(f"\n{'='*80}")
    print(f"Analyzing {split.upper()} split")
    print(f"{'='*80}")
    
    # Load data
    print(f"Loading HelpSteer3 '{split}' split...")
    dataset = load_dataset("nvidia/HelpSteer3", "feedback", split=split)
    data = list(dataset)
    
    if max_samples and len(data) > max_samples:
        random.seed(seed)
        data = random.sample(data, max_samples)
        print(f"Sampled {max_samples} examples from {len(dataset)} total")
    else:
        print(f"Loaded {len(data)} examples")
    
    # Storage for lengths
    case1_prompt_lengths = []
    case2_prompt_lengths = []
    response1_lengths = []
    response2_lengths = []
    context_lengths = []
    
    print("Processing examples...")
    for example in tqdm(data):
        context = example["context"]
        response1 = example.get("response1", "")
        response2 = example.get("response2", "")
        feedback1 = example.get("feedback1", [])
        feedback2 = example.get("feedback2", [])
        
        # Case 1: Just the conversation context
        case1_messages = create_case1_prompt(context)
        case1_text = tokenizer.apply_chat_template(
            case1_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        case1_tokens = tokenizer(case1_text, return_tensors="pt")
        case1_prompt_lengths.append(case1_tokens['input_ids'].shape[1])
        
        # Also get just context length (for reference)
        context_only_text = tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=False
        )
        context_tokens = tokenizer(context_only_text, return_tensors="pt")
        context_lengths.append(context_tokens['input_ids'].shape[1])
        
        # Case 2: With previous response and feedback
        # Randomly choose response1 or response2
        random.seed(seed + len(case2_prompt_lengths))
        if random.random() < 0.5:
            chosen_response = response1
            chosen_feedback = feedback1
        else:
            chosen_response = response2
            chosen_feedback = feedback2
        
        case2_messages = create_case2_prompt(context, chosen_response, chosen_feedback)
        case2_text = tokenizer.apply_chat_template(
            case2_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        case2_tokens = tokenizer(case2_text, return_tensors="pt")
        case2_prompt_lengths.append(case2_tokens['input_ids'].shape[1])
        
        # Response lengths
        if response1:
            response1_tokens = tokenizer(response1, return_tensors="pt")
            response1_lengths.append(response1_tokens['input_ids'].shape[1])
        
        if response2:
            response2_tokens = tokenizer(response2, return_tensors="pt")
            response2_lengths.append(response2_tokens['input_ids'].shape[1])
    
    # Compute statistics
    results = {
        "split": split,
        "num_examples": len(data),
        "context_only": compute_statistics(context_lengths, "Context Only (no chat template)"),
        "case1_prompt": compute_statistics(case1_prompt_lengths, "Case 1 Prompt (Context + Generation Prompt)"),
        "case2_prompt": compute_statistics(case2_prompt_lengths, "Case 2 Prompt (Context + Prev Response + Feedback + Generation Prompt)"),
        "response1": compute_statistics(response1_lengths, "Response1 Length"),
        "response2": compute_statistics(response2_lengths, "Response2 Length"),
    }
    
    # Print results
    print_statistics(results["context_only"])
    print_statistics(results["case1_prompt"])
    print_statistics(results["case2_prompt"])
    print_statistics(results["response1"])
    print_statistics(results["response2"])
    
    # Additional analysis
    print(f"\n{'='*80}")
    print("Additional Analysis")
    print(f"{'='*80}")
    
    # Overhead from chat template
    overhead = np.array(case1_prompt_lengths) - np.array(context_lengths)
    print(f"\nChat template overhead (Case 1 - Context):")
    print(f"  Mean: {np.mean(overhead):.1f} tokens")
    print(f"  Median: {np.median(overhead):.1f} tokens")
    
    # Case 2 expansion
    expansion = np.array(case2_prompt_lengths) - np.array(case1_prompt_lengths)
    print(f"\nCase 2 expansion (Case 2 - Case 1):")
    print(f"  Mean: {np.mean(expansion):.1f} tokens")
    print(f"  Median: {np.median(expansion):.1f} tokens")
    print(f"  Min: {np.min(expansion):,} tokens")
    print(f"  Max: {np.max(expansion):,} tokens")
    
    # Check how many exceed common limits
    limits = [512, 1024, 2048, 4096]
    print(f"\nPrompts exceeding common limits:")
    print(f"  Case 1:")
    for limit in limits:
        count = sum(1 for x in case1_prompt_lengths if x > limit)
        pct = count / len(case1_prompt_lengths) * 100
        print(f"    > {limit:,}: {count:,} ({pct:.1f}%)")
    
    print(f"  Case 2:")
    for limit in limits:
        count = sum(1 for x in case2_prompt_lengths if x > limit)
        pct = count / len(case2_prompt_lengths) * 100
        print(f"    > {limit:,}: {count:,} ({pct:.1f}%)")
    
    print(f"\nResponses exceeding common limits:")
    all_responses = response1_lengths + response2_lengths
    for limit in [256, 512, 768, 1024]:
        count = sum(1 for x in all_responses if x > limit)
        pct = count / len(all_responses) * 100
        print(f"  > {limit:,}: {count:,} ({pct:.1f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze HelpSteer3 dataset lengths")
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Splits to analyze (train, validation, test)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Tokenizer to use for length analysis",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to analyze per split (for faster analysis)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("HelpSteer3 Dataset Length Analysis")
    print("="*80)
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Splits: {', '.join(args.splits)}")
    if args.max_samples:
        print(f"Max samples per split: {args.max_samples}")
    print("="*80)
    
    # Load tokenizer
    print(f"\nLoading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Analyze each split
    all_results = {}
    for split in args.splits:
        # Map 'test' to validation since test is extracted from validation
        actual_split = "validation" if split == "test" else split
        results = analyze_split(
            actual_split,
            tokenizer,
            max_samples=args.max_samples,
            seed=args.seed
        )
        all_results[split] = results
    
    # Summary comparison across splits
    if len(args.splits) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY COMPARISON ACROSS SPLITS")
        print(f"{'='*80}")
        
        print(f"\n{'Split':<12} {'Case1 Mean':<12} {'Case1 P95':<12} {'Case2 Mean':<12} {'Case2 P95':<12}")
        print("-" * 80)
        for split in args.splits:
            case1_mean = all_results[split]["case1_prompt"]["mean"]
            case1_p95 = all_results[split]["case1_prompt"]["p95"]
            case2_mean = all_results[split]["case2_prompt"]["mean"]
            case2_p95 = all_results[split]["case2_prompt"]["p95"]
            print(f"{split:<12} {case1_mean:<12.1f} {case1_p95:<12.1f} {case2_mean:<12.1f} {case2_p95:<12.1f}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    # Use train split for recommendations (or first available)
    ref_split = "train" if "train" in all_results else list(all_results.keys())[0]
    ref_results = all_results[ref_split]
    
    case1_p95 = ref_results["case1_prompt"]["p95"]
    case2_p95 = ref_results["case2_prompt"]["p95"]
    response_p95 = max(
        ref_results["response1"]["p95"],
        ref_results["response2"]["p95"]
    )
    
    print(f"\nBased on {ref_split} split analysis:")
    print(f"\nCase 1 (Question Only):")
    print(f"  Recommended max_prompt_length: {int(np.ceil(case1_p95 / 128) * 128)} (covers 95% of prompts)")
    print(f"  Current config: 1024")
    if case1_p95 > 1024:
        print(f"  ⚠️  WARNING: {ref_results['case1_prompt']['p95']:.1f} > 1024, some prompts will be truncated!")
    else:
        print(f"  ✓ OK: Current config is sufficient")
    
    print(f"\nCase 2 (With Feedback):")
    print(f"  Recommended max_prompt_length: {int(np.ceil(case2_p95 / 128) * 128)} (covers 95% of prompts)")
    print(f"  Current config: 2048")
    if case2_p95 > 2048:
        print(f"  ⚠️  WARNING: {ref_results['case2_prompt']['p95']:.1f} > 2048, some prompts will be truncated!")
    else:
        print(f"  ✓ OK: Current config is sufficient")
    
    print(f"\nResponses:")
    print(f"  Recommended max_response_length: {int(np.ceil(response_p95 / 128) * 128)} (covers 95% of responses)")
    print(f"  Current config: 768")
    if response_p95 > 768:
        print(f"  ⚠️  WARNING: {response_p95:.1f} > 768, some responses will be truncated!")
    else:
        print(f"  ✓ OK: Current config is sufficient")
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

