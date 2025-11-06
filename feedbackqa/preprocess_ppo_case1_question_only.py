#!/usr/bin/env python3
"""
Case 1: Question Only (Baseline)
=================================

Preprocessing script for PPO training with QUESTION ONLY as input.

Input Format (feedback_train_ppo.json):
    {
        "question": "How do I get help finding a job?",
        "section_content": "In this rapidly changing jobs market...",
        "feedback": "A link to a job search website is included...",
        "rating": "Excellent"
    }

Output Format (parquet for verl):
    {
        "data_source": "feedback_qa",
        "prompt": [{"role": "user", "content": "How do I get help finding a job?"}],
        "ability": "qa_generation",
        "reward_model": {
            "style": "model",  # Use trained RM
            "ground_truth": "In this rapidly changing jobs market...",  # Reference (not used by RM)
            "feedback": "A link to a job search website is included...",
            "rating": "Excellent"
        },
        "extra_info": {...}
    }

PPO Training Flow:
    1. Model receives: "How do I get help finding a job?"
    2. Model generates: Its own answer
    3. RM scores: Question + Generated Answer
    4. Policy learns to maximize RM score

This is the BASELINE - model learns from scratch without seeing feedback examples.
"""

import argparse
import json
import os
import random
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from verl.utils.fs import copy, makedirs


def load_feedback_data(file_path: str) -> List[Dict]:
    """Load feedback QA data from JSON file"""
    print(f"Loading data from: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples")
    return data


def make_map_fn(split: str, data_source: str = "feedback_qa"):
    """
    Create mapping function for Case 1: Question Only
    
    The prompt contains ONLY the question.
    Model must generate answer from scratch.
    """
    def process_fn(example: Dict, idx: int) -> Dict:
        question = example["question"]
        section_content = example["section_content"]
        feedback = example.get("feedback", "")
        rating = example.get("rating", "")
        
        # Case 1: Input is ONLY the question
        prompt_messages = [
            {
                "role": "user",
                "content": question
            }
        ]
        
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "ability": "qa_generation",
            "reward_model": {
                "style": "model",  # Use trained neural network RM
                "ground_truth": section_content,  # Store for reference (not used by RM)
                "feedback": feedback,
                "rating": rating,
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "case": "question_only",
                "question": question,
            },
        }
        return data
    
    return process_fn


def preprocess_dataset(
    input_file: str,
    output_file: str,
    split: str,
    seed: int = 42,
) -> None:
    """Preprocess feedback QA data for Case 1 (Question Only)"""
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Load data
    raw_data = load_feedback_data(input_file)
    
    # Process each example
    processed_data = []
    map_fn = make_map_fn(split)
    
    print(f"Processing {len(raw_data)} examples for Case 1 (Question Only)...")
    for idx, example in enumerate(tqdm(raw_data)):
        processed = map_fn(example, idx)
        processed_data.append(processed)
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(processed_data)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save
    df.to_parquet(output_file)
    print(f"Saved {len(df)} examples to {output_file}")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {len(df)}")
    print(f"  Split: {split}")
    print(f"  Case: Question Only (Baseline)")
    print(f"  Data source: {df['data_source'].iloc[0]}")
    print(f"\nExample prompt:")
    print(f"  {df['prompt'].iloc[0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess feedback QA data for PPO training (Case 1: Question Only)"
    )
    
    parser.add_argument(
        "--train_file",
        type=str,
        default="feedbackqa/feedback_train_ppo.json",
        help="Path to training data JSON file",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default="feedbackqa/feedback_valid_ppo.json",
        help="Path to validation data JSON file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="feedbackqa/feedback_test_ppo.json",
        help="Path to test data JSON file",
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default="~/data/feedback_qa_ppo/case1_question_only",
        help="Local directory to save processed parquet files",
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Optional HDFS directory to copy files to",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Expand paths
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    # Process each split
    splits = [
        ("train", args.train_file, "train.parquet"),
        ("valid", args.valid_file, "valid.parquet"),
        ("test", args.test_file, "test.parquet"),
    ]
    
    for split_name, input_file, output_filename in splits:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping {split_name} split")
            continue
        
        output_file = os.path.join(local_save_dir, output_filename)
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        preprocess_dataset(
            input_file=input_file,
            output_file=output_file,
            split=split_name,
            seed=args.seed,
        )
        
        # Copy to HDFS if specified
        if args.hdfs_dir is not None:
            hdfs_path = os.path.join(args.hdfs_dir, output_filename)
            print(f"Copying to HDFS: {hdfs_path}")
            makedirs(os.path.dirname(hdfs_path))
            copy(output_file, hdfs_path)
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    print(f"\nProcessed files saved to: {local_save_dir}")
    print(f"\nTo use in PPO training:")
    print(f"  data.train_files={os.path.join(local_save_dir, 'train.parquet')}")
    print(f"  data.val_files={os.path.join(local_save_dir, 'valid.parquet')}")
    print(f"  reward_model.enable=True")
    print(f"  reward_model.model.path=./feedback_qa_reward_model/final_model")


if __name__ == "__main__":
    main()

