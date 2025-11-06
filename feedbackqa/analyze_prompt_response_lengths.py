#!/usr/bin/env python3
"""
Analyze prompt and response lengths for Case 1 vs Case 2.

This script analyzes:
1. Prompt lengths (with/without previous answer and feedback)
2. Response lengths (section_content)
3. Total sequence lengths (prompt + response)
4. Statistics and comparisons

Using tokenizer: meta-llama/Llama-3.1-8B-Instruct
"""

import json
import os
import numpy as np
from transformers import AutoTokenizer
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


def load_raw_data(file_path: str):
    """Load raw JSON data"""
    print(f"Loading raw data from: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} examples")
    return data


def analyze_raw_data_statistics(data, tokenizer):
    """Analyze statistics from raw data"""
    print("\n" + "="*80)
    print("RAW DATA ANALYSIS (feedback_train_ppo.json)")
    print("="*80)
    
    # Tokenize all questions, answers, and feedback
    question_lengths = []
    answer_lengths = []
    feedback_lengths = []
    
    # Count questions that appear multiple times
    question_counts = defaultdict(int)
    
    print("\nTokenizing all examples...")
    for example in tqdm(data):
        question = example["question"]
        answer = example["section_content"]
        feedback = example.get("feedback", "")
        
        question_counts[question] += 1
        
        # Tokenize
        q_tokens = tokenizer.encode(question, add_special_tokens=False)
        a_tokens = tokenizer.encode(answer, add_special_tokens=False)
        f_tokens = tokenizer.encode(feedback, add_special_tokens=False) if feedback else []
        
        question_lengths.append(len(q_tokens))
        answer_lengths.append(len(a_tokens))
        feedback_lengths.append(len(f_tokens))
    
    # Statistics
    print(f"\n{'Metric':<30} {'Min':>10} {'Mean':>10} {'Median':>10} {'90th %':>10} {'95th %':>10} {'Max':>10}")
    print("-" * 90)
    
    def print_stats(name, lengths):
        print(f"{name:<30} {np.min(lengths):>10.0f} {np.mean(lengths):>10.1f} "
              f"{np.median(lengths):>10.0f} {np.percentile(lengths, 90):>10.0f} "
              f"{np.percentile(lengths, 95):>10.0f} {np.max(lengths):>10.0f}")
    
    print_stats("Question length (tokens)", question_lengths)
    print_stats("Answer length (tokens)", answer_lengths)
    print_stats("Feedback length (tokens)", feedback_lengths)
    
    # Question reuse statistics
    unique_questions = len(question_counts)
    total_examples = len(data)
    questions_with_multiple = sum(1 for count in question_counts.values() if count > 1)
    avg_answers_per_question = np.mean(list(question_counts.values()))
    max_answers_per_question = max(question_counts.values())
    
    print(f"\n{'Question Reuse Statistics':<30}")
    print("-" * 50)
    print(f"{'Total examples':<30} {total_examples:>10}")
    print(f"{'Unique questions':<30} {unique_questions:>10}")
    print(f"{'Questions with >1 answer':<30} {questions_with_multiple:>10} ({questions_with_multiple/unique_questions*100:.1f}%)")
    print(f"{'Avg answers per question':<30} {avg_answers_per_question:>10.2f}")
    print(f"{'Max answers per question':<30} {max_answers_per_question:>10}")
    
    return question_lengths, answer_lengths, feedback_lengths, question_counts


def analyze_case1_prompts(data, tokenizer):
    """Analyze Case 1 prompts (question only)"""
    print("\n" + "="*80)
    print("CASE 1: QUESTION ONLY (Baseline)")
    print("="*80)
    
    prompt_lengths = []
    
    print("\nCreating Case 1 prompts...")
    for example in tqdm(data):
        question = example["question"]
        
        # Case 1 format: just the question
        prompt = question
        
        # Tokenize
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_lengths.append(len(prompt_tokens))
    
    print(f"\n{'Metric':<30} {'Min':>10} {'Mean':>10} {'Median':>10} {'90th %':>10} {'95th %':>10} {'Max':>10}")
    print("-" * 90)
    print(f"{'Prompt length (tokens)':<30} {np.min(prompt_lengths):>10.0f} {np.mean(prompt_lengths):>10.1f} "
          f"{np.median(prompt_lengths):>10.0f} {np.percentile(prompt_lengths, 90):>10.0f} "
          f"{np.percentile(prompt_lengths, 95):>10.0f} {np.max(prompt_lengths):>10.0f}")
    
    return prompt_lengths


def analyze_case2_prompts(data, tokenizer):
    """Analyze Case 2 prompts (with previous answer and feedback)"""
    print("\n" + "="*80)
    print("CASE 2: WITH PREVIOUS ANSWER + FEEDBACK (Experimental)")
    print("="*80)
    
    # Build question index
    question_to_examples = defaultdict(list)
    for idx, example in enumerate(data):
        question_to_examples[example["question"]].append(idx)
    
    prompt_lengths = []
    prompt_lengths_with_context = []
    prompt_lengths_no_context = []
    examples_with_context = 0
    examples_without_context = 0
    
    print("\nCreating Case 2 prompts...")
    for idx, example in enumerate(tqdm(data)):
        question = example["question"]
        
        # Find other examples with same question
        same_question_indices = [i for i in question_to_examples[question] if i != idx]
        
        if same_question_indices:
            # Has previous answer - Case 2 format
            prev_idx = same_question_indices[0]  # Take first one for analysis
            prev_example = data[prev_idx]
            
            previous_answer = prev_example["section_content"]
            previous_feedback = prev_example.get("feedback", "")
            previous_rating = prev_example.get("rating", "")
            
            prompt = (
                f"Here is a previous answer to this question with feedback:\n\n"
                f"Question: {question}\n\n"
                f"Previous Answer: {previous_answer}\n\n"
                f"Feedback: {previous_feedback} (Rating: {previous_rating})\n\n"
                f"Now, please answer the same question:\n\n"
                f"Question: {question}"
            )
            
            examples_with_context += 1
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_lengths_with_context.append(len(prompt_tokens))
        else:
            # No previous answer - fallback to Case 1
            prompt = question
            examples_without_context += 1
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_lengths_no_context.append(len(prompt_tokens))
        
        prompt_lengths.append(len(prompt_tokens))
    
    print(f"\n{'Distribution':<30}")
    print("-" * 50)
    print(f"{'Examples WITH context':<30} {examples_with_context:>10} ({examples_with_context/len(data)*100:.1f}%)")
    print(f"{'Examples WITHOUT context':<30} {examples_without_context:>10} ({examples_without_context/len(data)*100:.1f}%)")
    
    print(f"\n{'Metric':<30} {'Min':>10} {'Mean':>10} {'Median':>10} {'90th %':>10} {'95th %':>10} {'Max':>10}")
    print("-" * 90)
    print(f"{'Overall prompt length':<30} {np.min(prompt_lengths):>10.0f} {np.mean(prompt_lengths):>10.1f} "
          f"{np.median(prompt_lengths):>10.0f} {np.percentile(prompt_lengths, 90):>10.0f} "
          f"{np.percentile(prompt_lengths, 95):>10.0f} {np.max(prompt_lengths):>10.0f}")
    
    if prompt_lengths_with_context:
        print(f"{'Prompt (WITH context)':<30} {np.min(prompt_lengths_with_context):>10.0f} {np.mean(prompt_lengths_with_context):>10.1f} "
              f"{np.median(prompt_lengths_with_context):>10.0f} {np.percentile(prompt_lengths_with_context, 90):>10.0f} "
              f"{np.percentile(prompt_lengths_with_context, 95):>10.0f} {np.max(prompt_lengths_with_context):>10.0f}")
    
    if prompt_lengths_no_context:
        print(f"{'Prompt (NO context)':<30} {np.min(prompt_lengths_no_context):>10.0f} {np.mean(prompt_lengths_no_context):>10.1f} "
              f"{np.median(prompt_lengths_no_context):>10.0f} {np.percentile(prompt_lengths_no_context, 90):>10.0f} "
              f"{np.percentile(prompt_lengths_no_context, 95):>10.0f} {np.max(prompt_lengths_no_context):>10.0f}")
    
    return prompt_lengths, prompt_lengths_with_context, prompt_lengths_no_context


def compare_cases(case1_prompts, case2_prompts, case2_with_context, answer_lengths):
    """Compare the two cases"""
    print("\n" + "="*80)
    print("COMPARISON: CASE 1 vs CASE 2")
    print("="*80)
    
    # Prompt length comparison
    print(f"\n{'Prompt Length Comparison':<40} {'Case 1':>15} {'Case 2':>15} {'Difference':>15}")
    print("-" * 85)
    print(f"{'Mean prompt length':<40} {np.mean(case1_prompts):>15.1f} {np.mean(case2_prompts):>15.1f} {np.mean(case2_prompts) - np.mean(case1_prompts):>15.1f}")
    print(f"{'Median prompt length':<40} {np.median(case1_prompts):>15.0f} {np.median(case2_prompts):>15.0f} {np.median(case2_prompts) - np.median(case1_prompts):>15.0f}")
    print(f"{'95th percentile':<40} {np.percentile(case1_prompts, 95):>15.0f} {np.percentile(case2_prompts, 95):>15.0f} {np.percentile(case2_prompts, 95) - np.percentile(case1_prompts, 95):>15.0f}")
    
    if case2_with_context:
        print(f"\n{'Case 2 WITH context only':<40} {'':>15} {'Case 2':>15} {'vs Case 1':>15}")
        print("-" * 85)
        print(f"{'Mean prompt length':<40} {'':>15} {np.mean(case2_with_context):>15.1f} {np.mean(case2_with_context) - np.mean(case1_prompts):>15.1f}")
        print(f"{'Additional tokens from context':<40} {'':>15} {'':>15} {np.mean(case2_with_context) - np.mean(case1_prompts):>15.1f}")
    
    # Total sequence length (prompt + response)
    print(f"\n{'Total Sequence Length (Prompt + Response)':<40} {'Case 1':>15} {'Case 2':>15} {'Difference':>15}")
    print("-" * 85)
    
    case1_total = [p + r for p, r in zip(case1_prompts, answer_lengths)]
    case2_total = [p + r for p, r in zip(case2_prompts, answer_lengths)]
    
    print(f"{'Mean total length':<40} {np.mean(case1_total):>15.1f} {np.mean(case2_total):>15.1f} {np.mean(case2_total) - np.mean(case1_total):>15.1f}")
    print(f"{'Median total length':<40} {np.median(case1_total):>15.0f} {np.median(case2_total):>15.0f} {np.median(case2_total) - np.median(case1_total):>15.0f}")
    print(f"{'95th percentile':<40} {np.percentile(case1_total, 95):>15.0f} {np.percentile(case2_total, 95):>15.0f} {np.percentile(case2_total, 95) - np.percentile(case1_total, 95):>15.0f}")
    print(f"{'Max total length':<40} {np.max(case1_total):>15.0f} {np.max(case2_total):>15.0f} {np.max(case2_total) - np.max(case1_total):>15.0f}")


def generate_summary_recommendations(case1_prompts, case2_prompts, case2_with_context, answer_lengths):
    """Generate recommendations for training configuration"""
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR TRAINING CONFIGURATION")
    print("="*80)
    
    case1_mean = np.mean(case1_prompts)
    case2_mean = np.mean(case2_prompts)
    case2_ctx_mean = np.mean(case2_with_context) if case2_with_context else case2_mean
    answer_mean = np.mean(answer_lengths)
    
    case1_95th = np.percentile(case1_prompts, 95)
    case2_95th = np.percentile(case2_prompts, 95)
    answer_95th = np.percentile(answer_lengths, 95)
    
    print(f"\n1. Maximum Prompt Length:")
    print(f"   Case 1: Set max_prompt_length >= {int(case1_95th)} (covers 95% of examples)")
    print(f"   Case 2: Set max_prompt_length >= {int(case2_95th)} (covers 95% of examples)")
    print(f"   Recommended: {int(max(case2_95th, 1024))} tokens")
    
    print(f"\n2. Maximum Response Length:")
    print(f"   Response 95th percentile: {int(answer_95th)} tokens")
    print(f"   Recommended: max_response_length >= {int(answer_95th)}")
    
    case1_total_95th = np.percentile([p + r for p, r in zip(case1_prompts, answer_lengths)], 95)
    case2_total_95th = np.percentile([p + r for p, r in zip(case2_prompts, answer_lengths)], 95)
    
    print(f"\n3. Total Sequence Length (for memory planning):")
    print(f"   Case 1: {int(case1_total_95th)} tokens (95th percentile)")
    print(f"   Case 2: {int(case2_total_95th)} tokens (95th percentile)")
    
    print(f"\n4. Context Window Increase:")
    print(f"   Case 2 needs ~{int(case2_ctx_mean - case1_mean)} more tokens on average")
    print(f"   This is {((case2_ctx_mean - case1_mean) / case1_mean * 100):.1f}% increase in prompt length")
    
    print(f"\n5. Batch Size Recommendations:")
    print(f"   Case 1: Can use larger batch sizes (shorter prompts)")
    print(f"   Case 2: May need 30-50% smaller batch size due to longer prompts")
    print(f"   Consider: Case1 batch=256, Case2 batch=128")
    
    print(f"\n6. Training Configuration:")
    print(f"   ```bash")
    print(f"   # Case 1")
    print(f"   data.max_prompt_length={int(max(case1_95th * 1.1, 512))}")
    print(f"   data.max_response_length={int(max(answer_95th * 1.1, 512))}")
    print(f"   data.train_batch_size=256")
    print(f"   ")
    print(f"   # Case 2")
    print(f"   data.max_prompt_length={int(max(case2_95th * 1.1, 1024))}")
    print(f"   data.max_response_length={int(max(answer_95th * 1.1, 512))}")
    print(f"   data.train_batch_size=128  # Reduced due to longer prompts")
    print(f"   ```")


def main():
    """Main analysis function"""
    print("="*80)
    print("PROMPT AND RESPONSE LENGTH ANALYSIS")
    print("Tokenizer: meta-llama/Llama-3.1-8B-Instruct")
    print("="*80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            trust_remote_code=True,
            use_fast=True
        )
        print(f"✓ Tokenizer loaded: {tokenizer.__class__.__name__}")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        print("\nTrying alternative tokenizer...")
        try:
            # Fallback to Llama-2 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                trust_remote_code=True,
                use_fast=True
            )
            print(f"✓ Using fallback tokenizer: {tokenizer.__class__.__name__}")
        except Exception as e2:
            print(f"❌ Error loading fallback tokenizer: {e2}")
            print("\nPlease ensure you have the transformers library installed:")
            print("  pip install transformers")
            return
    
    # Load raw data
    data_file = "feedbackqa/feedback_train_ppo.json"
    if not os.path.exists(data_file):
        print(f"\n❌ Error: {data_file} not found!")
        print("Please ensure the data file exists.")
        return
    
    data = load_raw_data(data_file)
    
    # Analyze raw data
    question_lengths, answer_lengths, feedback_lengths, question_counts = analyze_raw_data_statistics(data, tokenizer)
    
    # Analyze Case 1
    case1_prompts = analyze_case1_prompts(data, tokenizer)
    
    # Analyze Case 2
    case2_prompts, case2_with_context, case2_no_context = analyze_case2_prompts(data, tokenizer)
    
    # Compare
    compare_cases(case1_prompts, case2_prompts, case2_with_context, answer_lengths)
    
    # Generate recommendations
    generate_summary_recommendations(case1_prompts, case2_prompts, case2_with_context, answer_lengths)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("1. Case 2 prompts are significantly longer due to previous answer context")
    print("2. Adjust max_prompt_length accordingly for each case")
    print("3. Consider reducing batch size for Case 2 to fit in memory")
    print("4. Response lengths are the same for both cases (same answers)")
    print("5. Most questions have multiple answers → good for Case 2!")


if __name__ == "__main__":
    main()

