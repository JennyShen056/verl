import argparse
import json
import os
import random
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from verl.utils.fs import copy, makedirs


def load_helpsteer3_data(split: str = "train", test_size: int = 500, seed: int = 42) -> List[Dict]:
    """
    Load HelpSteer3 feedback subset from HuggingFace
    
    Args:
        split: 'train', 'validation', or 'test'
        test_size: Number of samples to reserve for test from validation
        seed: Random seed for reproducible train/val/test split
    
    Returns:
        List of examples for the requested split
    """
    if split == "test":
        # Load validation and extract test samples
        print(f"Loading HelpSteer3 'feedback' subset, extracting test from validation")
        dataset = load_dataset("nvidia/HelpSteer3", "feedback", split="validation")
        all_data = list(dataset)
        
        # Use seed to get consistent test indices
        random.seed(seed)
        test_indices = set(random.sample(range(len(all_data)), test_size))
        test_data = [all_data[i] for i in sorted(test_indices)]
        
        print(f"Loaded {len(test_data)} test examples (split from validation)")
        return test_data
    
    elif split == "validation":
        # Load validation and exclude test samples
        print(f"Loading HelpSteer3 'feedback' subset, validation (excluding test)")
        dataset = load_dataset("nvidia/HelpSteer3", "feedback", split="validation")
        all_data = list(dataset)
        
        # Use seed to get consistent test indices to exclude
        random.seed(seed)
        test_indices = set(random.sample(range(len(all_data)), test_size))
        validation_data = [all_data[i] for i in range(len(all_data)) if i not in test_indices]
        
        print(f"Loaded {len(validation_data)} validation examples (originally {len(all_data)}, reserved {test_size} for test)")
        return validation_data
    
    else:  # train
        print(f"Loading HelpSteer3 'feedback' subset, split: {split}")
        dataset = load_dataset("nvidia/HelpSteer3", "feedback", split=split)
        print(f"Loaded {len(dataset)} examples")
        return list(dataset)


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


def make_map_fn(split: str, data_source: str = "helpsteer3_feedback"):
    """
    Create mapping function for Case 1: Question Only
    
    The prompt contains ONLY the context (conversation).
    Model must generate response from scratch.
    """
    def process_fn(example: Dict, idx: int) -> Dict:
        # Extract context (conversation) as the question
        context = example["context"]
        
        # Format the conversation into a readable prompt
        formatted_context = format_conversation(context)
        
        # Get metadata
        domain = example.get("domain", "")
        language = example.get("language", "")
        
        # Case 1: Input is ONLY the context (conversation)
        # Use the conversation list as-is for the prompt
        prompt_messages = context.copy()
        
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "ability": "qa_generation",
            "reward_model": {
                "style": "model",  # Use trained neural network RM
                "ground_truth": None,  # HelpSteer3 doesn't have ground truth answers
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "case": "question_only",
                "domain": domain,
                "language": language,
                "conversation_length": len(context),
            },
        }
        return data
    
    return process_fn


def preprocess_dataset(
    split: str,
    output_file: str,
    seed: int = 42,
    test_size: int = 500,
) -> None:
    """Preprocess HelpSteer3 feedback data for Case 1 (Question Only)"""
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Load data (with test split handling)
    raw_data = load_helpsteer3_data(split, test_size=test_size, seed=seed)
    
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
    print(f"\nExample prompt (first message):")
    print(f"  {df['prompt'].iloc[0][0]}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HelpSteer3 feedback data for PPO training (Case 1: Question Only)"
    )
    
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="Splits to process (e.g., train validation test)",
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default="~/data/helpsteer3_ppo/case1_question_only",
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
    parser.add_argument(
        "--test_size",
        type=int,
        default=500,
        help="Number of samples to reserve for test from validation split",
    )
    
    args = parser.parse_args()
    
    # Expand paths
    local_save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)
    
    # Process each split
    for split_name in args.splits:
        output_filename = f"{split_name}.parquet"
        output_file = os.path.join(local_save_dir, output_filename)
        
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        try:
            preprocess_dataset(
                split=split_name,
                output_file=output_file,
                seed=args.seed,
                test_size=args.test_size,
            )
            
            # Copy to HDFS if specified
            if args.hdfs_dir is not None:
                hdfs_path = os.path.join(args.hdfs_dir, output_filename)
                print(f"Copying to HDFS: {hdfs_path}")
                makedirs(os.path.dirname(hdfs_path))
                copy(output_file, hdfs_path)
        except Exception as e:
            print(f"Error processing {split_name} split: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"{'='*60}")
    print(f"\nProcessed files saved to: {local_save_dir}")
    print(f"\nTo use in PPO training:")
    print(f"  data.train_files={os.path.join(local_save_dir, 'train.parquet')}")
    print(f"  data.val_files={os.path.join(local_save_dir, 'validation.parquet')}")
    print(f"  data.test_files={os.path.join(local_save_dir, 'test.parquet')}")
    print(f"  reward_model.enable=True")
    print(f"  reward_model.model.path=./helpsteer3_reward_model/final_model")


if __name__ == "__main__":
    main()

