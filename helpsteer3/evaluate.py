"""
Evaluation script for HelpSteer3 predictions using the reward model.
Uses Jennny/llama3_help_rm for scoring.
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline


def load_predictions(predictions_file: str):
    """Load predictions from JSON file"""
    print(f"Loading predictions from {predictions_file}...")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions")
    return predictions


def evaluate_with_reward_model(
    predictions,
    reward_model_path: str = "Jennny/llama3_help_rm",
    batch_size: int = 8,
):
    """
    Evaluate predictions using the reward model.
    
    Args:
        predictions: List of prediction dicts with 'context' and 'generated_response'
        reward_model_path: Path to reward model (uses FsfairX-LLaMA3-RM-v0.1)
        batch_size: Batch size for evaluation
    
    Returns:
        List of predictions with added 'reward_score' field
    """
    
    print(f"\nLoading reward model: {reward_model_path}")
    
    # Load tokenizer and pipeline
    rm_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)
    
    # Set pad_token if not already set (required for batching)
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id
        print(f"Set pad_token to eos_token for batching")
    
    rm_pipe = pipeline(
        "sentiment-analysis",
        model=reward_model_path,
        device=0 if torch.cuda.is_available() else -1,
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    
    pipe_kwargs = {
        "top_k": None,  # Use top_k instead of deprecated return_all_scores
        "function_to_apply": "none",
        "batch_size": batch_size
    }
    
    print(f"Evaluating {len(predictions)} predictions...")
    
    results = []
    
    for i in tqdm(range(0, len(predictions), batch_size), desc="Computing rewards"):
        batch = predictions[i:i + batch_size]
        
        # Prepare chats for reward model
        chats = []
        for pred in batch:
            # Create chat with context + generated response
            chat = pred["context"].copy()
            chat.append({
                "role": "assistant",
                "content": pred["generated_response"]
            })
            chats.append(chat)
        
        # Convert to text format (without BOS token as per the example)
        test_texts = [
            rm_tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=False
            ).replace(rm_tokenizer.bos_token, "")
            for chat in chats
        ]
        
        # Get rewards
        pipe_outputs = rm_pipe(test_texts, **pipe_kwargs)
        rewards = [output[0]["score"] for output in pipe_outputs]
        
        # Add rewards to results
        for pred, reward in zip(batch, rewards):
            pred_with_reward = pred.copy()
            pred_with_reward["reward_score"] = reward
            results.append(pred_with_reward)
    
    return results


def compute_statistics(results):
    """Compute evaluation statistics"""
    rewards = [r["reward_score"] for r in results]
    
    stats = {
        "num_examples": len(results),
        "mean_reward": sum(rewards) / len(rewards),
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "median_reward": sorted(rewards)[len(rewards) // 2],
    }
    
    # Compute by domain if available
    domains = {}
    for r in results:
        domain = r.get("domain", "unknown")
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(r["reward_score"])
    
    stats["by_domain"] = {
        domain: {
            "count": len(scores),
            "mean_reward": sum(scores) / len(scores)
        }
        for domain, scores in domains.items()
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate HelpSteer3 predictions with reward model")
    parser.add_argument("--predictions_file", type=str, required=True, help="Input predictions JSON")
    parser.add_argument("--reward_model_path", type=str, default="Jennny/llama3_help_rm",
                       help="Path to reward model")
    parser.add_argument("--output_file", type=str, required=True, help="Output evaluation JSON")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    
    args = parser.parse_args()
    
    print("="*80)
    print("HelpSteer3 Evaluation with Reward Model")
    print("="*80)
    print(f"Predictions: {args.predictions_file}")
    print(f"Reward Model: {args.reward_model_path}")
    print(f"Output: {args.output_file}")
    print("="*80)
    
    # Load predictions
    predictions = load_predictions(args.predictions_file)
    
    # Evaluate with reward model
    results = evaluate_with_reward_model(
        predictions,
        reward_model_path=args.reward_model_path,
        batch_size=args.batch_size,
    )
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(results)
    
    # Print statistics
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Number of examples: {stats['num_examples']}")
    print(f"Mean reward: {stats['mean_reward']:.4f}")
    print(f"Median reward: {stats['median_reward']:.4f}")
    print(f"Min reward: {stats['min_reward']:.4f}")
    print(f"Max reward: {stats['max_reward']:.4f}")
    
    print("\nRewards by domain:")
    for domain, domain_stats in stats['by_domain'].items():
        print(f"  {domain}: {domain_stats['mean_reward']:.4f} (n={domain_stats['count']})")
    
    # Save results
    output_data = {
        "statistics": stats,
        "predictions": results
    }
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Saved evaluation results to {args.output_file}")


if __name__ == "__main__":
    main()

