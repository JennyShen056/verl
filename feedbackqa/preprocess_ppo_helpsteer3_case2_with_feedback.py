import argparse
import json
import os
import random
from typing import Dict, List

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from verl.utils.fs import copy, makedirs


def load_helpsteer3_data(split: str = "train") -> List[Dict]:
    """Load HelpSteer3 feedback subset from HuggingFace"""
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


def format_feedback_list(feedback_list: List[str]) -> str:
    """Convert feedback list to a formatted string"""
    if not feedback_list:
        return "No specific feedback provided."
    return "\n".join([f"- {fb}" for fb in feedback_list])


def make_map_fn(split: str, data_source: str = "helpsteer3_feedback", seed: int = 42):
    """
    Create mapping function for Case 2: Same Conversation + Previous Response + Feedback (Single-Turn)
    
    Similar to FeedbackQA format - everything in one prompt string.
    The prompt shows a PREVIOUS response to the SAME conversation with feedback,
    then asks the model to respond to that same conversation.
    
    For each example, randomly choose response1+feedback1 OR response2+feedback2.
    Model learns from seeing example responses to the exact conversation with feedback.
    """
    def process_fn(example: Dict, idx: int) -> Dict:
        # Set seed for this specific example to ensure reproducibility
        random.seed(seed + idx)
        
        # Extract context (conversation) as the question
        context = example["context"]
        formatted_context = format_conversation(context)
        
        # Get metadata
        domain = example.get("domain", "")
        language = example.get("language", "")
        
        # Randomly choose response1 or response2 (and corresponding feedback)
        response1 = example.get("response1", "")
        response2 = example.get("response2", "")
        feedback1 = example.get("feedback1", [])
        feedback2 = example.get("feedback2", [])
        
        # Choose randomly between the two responses
        if random.random() < 0.5:
            chosen_response = response1
            chosen_feedback = feedback1
            chosen_response_id = 1
        else:
            chosen_response = response2
            chosen_feedback = feedback2
            chosen_response_id = 2
        
        # Format feedback
        formatted_feedback = format_feedback_list(chosen_feedback)
        
        # Case 2: Single-turn prompt with previous response to SAME context
        # Mimicking the FeedbackQA format: show context, previous response with feedback, then ask to respond
        prompt_content = (
            f"Here is a previous response to this conversation with feedback:\n\n"
            f"Conversation:\n{formatted_context}\n\n"
            f"Previous Response: {chosen_response}\n\n"
            f"Feedback: {formatted_feedback}\n\n"
            f"Now, please respond to the same conversation:\n\n"
            f"Conversation:\n{formatted_context}"
        )
        
        # Single-turn prompt (like FeedbackQA case2)
        prompt_messages = [
            {
                "role": "user",
                "content": prompt_content
            }
        ]
        
        # Question-only prompt for reward model (without previous response/feedback context)
        question_only_prompt = context.copy()
        
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,  # Full context for policy training
            "ability": "qa_generation",
            "reward_model": {
                "style": "model",  # Use trained neural network RM
                "ground_truth": None,  # HelpSteer3 doesn't have ground truth answers
                "prompt_for_rm": question_only_prompt,  # Context-only for RM evaluation
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "case": "with_previous_response_and_feedback",
                "domain": domain,
                "language": language,
                "conversation_length": len(context),
                "chosen_response_id": chosen_response_id,
                "previous_response": chosen_response,
                "previous_feedback": chosen_feedback,
            },
        }
        return data
    
    return process_fn


def preprocess_dataset(
    split: str,
    output_file: str,
    seed: int = 42,
) -> None:
    """Preprocess HelpSteer3 feedback data for Case 2 (With Feedback)"""
    
    # Set seed for reproducibility
    random.seed(seed)
    
    # Load data
    raw_data = load_helpsteer3_data(split)
    
    # Process each example
    processed_data = []
    map_fn = make_map_fn(split, seed=seed)
    
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
    print(f"  Case: With Previous Response and Feedback (Feedback-Augmented)")
    print(f"  Data source: {df['data_source'].iloc[0]}")
    
    # Count response selection distribution
    response1_count = df['extra_info'].apply(lambda x: x['chosen_response_id'] == 1).sum()
    response2_count = df['extra_info'].apply(lambda x: x['chosen_response_id'] == 2).sum()
    print(f"  Response1 selected: {response1_count} ({response1_count/len(df)*100:.1f}%)")
    print(f"  Response2 selected: {response2_count} ({response2_count/len(df)*100:.1f}%)")
    
    # Show example
    print(f"\nExample prompt (with previous response to same conversation):")
    example_prompt = df['prompt'].iloc[0]['content']
    # Show first 600 chars
    print(f"  {example_prompt[:600]}...")
    print(f"\nPrompt structure: Previous Response to SAME Conversation + Feedback → Respond to Conversation Again")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess HelpSteer3 feedback data for PPO training (Case 2: With Feedback)"
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
        default="~/data/helpsteer3_ppo/case2_with_feedback",
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
    print(f"  reward_model.enable=True")
    print(f"  reward_model.model.path=./helpsteer3_reward_model/final_model")


if __name__ == "__main__":
    main()

