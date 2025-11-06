import argparse
import json
import os
import random
from typing import Dict, List, Optional

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


def sample_previous_answer_for_same_question(
    all_examples: List[Dict],
    current_question: str,
    current_idx: int,
    seed: Optional[int] = None,
) -> Optional[Dict]:
    """
    Sample a previous answer to THE SAME question with feedback.
    
    This shows the model an example answer to the exact question it needs to answer,
    teaching it what makes a good answer to THIS SPECIFIC question.
    
    Args:
        all_examples: Full dataset
        current_question: The question we're trying to answer
        current_idx: Current example index (to avoid self-reference)
        seed: Random seed for reproducibility
        
    Returns:
        Example with same question but different answer/feedback, or None if no match
    """
    if seed is not None:
        random.seed(seed + current_idx)  # Different seed per example
    
    # Find all examples with the SAME question but different index
    same_question_indices = [
        i for i in range(len(all_examples))
        if i != current_idx and all_examples[i]["question"] == current_question
    ]
    
    # If no other examples for this question, return None
    if not same_question_indices:
        return None
    
    # Randomly select one of the matching examples
    random_idx = random.choice(same_question_indices)
    
    return all_examples[random_idx]


def make_map_fn(split: str, all_examples: List[Dict], data_source: str = "feedback_qa"):
    """
    Create mapping function for Case 2: Same Question + Previous Answer + Feedback (Single-Turn)
    
    Similar to gsm8k.py format - everything in one prompt string.
    The prompt shows a PREVIOUS answer to the SAME question with feedback,
    then asks the model to answer that same question.
    
    Model learns from seeing example answers to the exact question with feedback.
    """
    def process_fn(example: Dict, idx: int) -> Dict:
        # Current example (the one to answer)
        question = example["question"]
        section_content = example["section_content"]
        feedback = example.get("feedback", "")
        rating = example.get("rating", "")
        
        # Try to find a previous answer to THE SAME question
        previous_example = sample_previous_answer_for_same_question(
            all_examples, question, idx, seed=42
        )
        
        if previous_example is not None:
            # Found a previous answer to the same question - use it as context
            previous_answer = previous_example["section_content"]
            previous_feedback = previous_example.get("feedback", "")
            previous_rating = previous_example.get("rating", "")
            
            # Case 2: Single-turn prompt with previous answer to SAME question
            prompt_content = (
                f"Here is a previous answer to this question with feedback:\n\n"
                f"Question: {question}\n\n"
                f"Previous Answer: {previous_answer}\n\n"
                f"Feedback: {previous_feedback} (Rating: {previous_rating})\n\n"
                f"Now, please answer the same question:\n\n"
                f"Question: {question}"
            )
            
            has_example = True
        else:
            # No other example for this question - just ask the question (fallback to Case 1)
            prompt_content = question
            has_example = False
            previous_feedback = None
            previous_rating = None
        
        # Single-turn prompt (like gsm8k.py)
        prompt_messages = [
            {
                "role": "user",
                "content": prompt_content
            }
        ]
        
        # Question-only prompt for reward model (without previous answer/feedback context)
        question_only_prompt = [
            {
                "role": "user",
                "content": question
            }
        ]
        
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,  # Full context for policy training
            "ability": "qa_generation",
            "reward_model": {
                "style": "model",  # Use trained neural network RM
                "ground_truth": section_content,  # Store for reference
                "feedback": feedback,
                "rating": rating,
                "prompt_for_rm": question_only_prompt,  # Question-only for RM evaluation
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "case": "with_previous_answer_same_question",
                "question": question,
                "has_previous_example": has_example,
                "previous_feedback": previous_feedback if has_example else None,
                "previous_rating": previous_rating if has_example else None,
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
    """Preprocess feedback QA data for Case 2 (With Feedback Examples)"""
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Load all data (needed for sampling random examples)
    raw_data = load_feedback_data(input_file)
    
    # Process each example
    processed_data = []
    map_fn = make_map_fn(split, raw_data)
    
    print(f"Processing {len(raw_data)} examples for Case 2 (With Feedback)...")
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
    print(f"  Case: With Previous Answer to Same Question (Feedback-Augmented)")
    print(f"  Data source: {df['data_source'].iloc[0]}")
    
    # Count how many have previous examples
    has_previous = df['extra_info'].apply(lambda x: x['has_previous_example']).sum()
    print(f"  Examples with previous answer: {has_previous} ({has_previous/len(df)*100:.1f}%)")
    print(f"  Examples without previous answer: {len(df) - has_previous} ({(len(df)-has_previous)/len(df)*100:.1f}%)")
    
    # Show example with previous answer
    example_with_prev = df[df['extra_info'].apply(lambda x: x['has_previous_example'])].iloc[0] if has_previous > 0 else None
    if example_with_prev is not None:
        print(f"\nExample prompt (with previous answer to same question):")
        example_prompt = example_with_prev['prompt'][0]['content']
        # Show first 600 chars
        print(f"  {example_prompt[:600]}...")
        print(f"\nPrompt structure: Previous Answer to SAME Question + Feedback → Answer Question Again")
    else:
        print(f"\nNo examples found with previous answers - all questions are unique!")
        print(f"Example prompt (fallback to Case 1):")
        print(f"  {df['prompt'].iloc[0][0]['content']}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess feedback QA data for PPO training (Case 2: With Feedback)"
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
        default="~/data/feedback_qa_ppo/case2_with_feedback",
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

