#!/usr/bin/env python3
"""
Evaluation Script for PPO Model Predictions
============================================

Evaluate generated answers using the reward model.

Usage:
    python feedbackqa/evaluate.py \
        --predictions_file outputs/case1_predictions.json \
        --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
        --output_file outputs/case1_evaluation.json
"""

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_reward_model(model_path: str, device: str = "auto"):
    """Load reward model for evaluation"""
    print(f"Loading reward model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    model.eval()
    
    print(f"✓ Reward model loaded")
    print(f"  Type: {model.__class__.__name__}")
    print(f"  Device: {next(model.parameters()).device}")
    
    return model, tokenizer


def load_predictions(predictions_file: str) -> List[Dict]:
    """Load model predictions"""
    print(f"Loading predictions from: {predictions_file}")
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    print(f"✓ Loaded {len(predictions)} predictions")
    return predictions


def format_for_reward_model(question: str, answer: str, tokenizer) -> str:
    """Format Q+A pair for reward model"""
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback format
        text = f"Question: {question}\n\nAnswer: {answer}"
    
    return text


def compute_reward_scores(
    reward_model,
    tokenizer,
    predictions: List[Dict],
    batch_size: int = 8,
) -> List[float]:
    """Compute reward scores for all predictions"""
    
    print(f"\nComputing reward scores...")
    print(f"  Batch size: {batch_size}")
    print(f"  Total predictions: {len(predictions)}")
    
    all_scores = []
    
    for i in tqdm(range(0, len(predictions), batch_size), desc="Scoring"):
        batch = predictions[i:i+batch_size]
        
        # Format for reward model
        texts = [
            format_for_reward_model(
                pred["question"],
                pred["generated_answer"],
                tokenizer
            )
            for pred in batch
        ]
        
        # Tokenize
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(reward_model.device)
        
        # Get scores
        with torch.no_grad():
            outputs = reward_model(**inputs)
            # For sequence classification, take the positive class logit
            # Assuming label 1 = good, label 0 = bad
            if outputs.logits.shape[-1] == 2:
                # Binary classification: use class 1 score
                scores = outputs.logits[:, 1].cpu().numpy()
            else:
                # Single output: use as-is
                scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        all_scores.extend(scores.tolist())
    
    return all_scores


def compute_metrics(scores: List[float]) -> Dict:
    """Compute evaluation metrics"""
    scores_array = np.array(scores)
    
    metrics = {
        "mean_reward": float(np.mean(scores_array)),
        "std_reward": float(np.std(scores_array)),
        "min_reward": float(np.min(scores_array)),
        "max_reward": float(np.max(scores_array)),
        "median_reward": float(np.median(scores_array)),
        "q25_reward": float(np.percentile(scores_array, 25)),
        "q75_reward": float(np.percentile(scores_array, 75)),
        "num_samples": len(scores),
    }
    
    return metrics


def save_evaluation_results(
    predictions: List[Dict],
    scores: List[float],
    metrics: Dict,
    output_file: str
):
    """Save evaluation results"""
    
    # Add scores to predictions
    results = []
    for pred, score in zip(predictions, scores):
        result = pred.copy()
        result["reward_score"] = float(score)
        results.append(result)
    
    # Create output
    output = {
        "metrics": metrics,
        "predictions": results
    }
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to: {output_file}")


def print_metrics(metrics: Dict):
    """Print evaluation metrics"""
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    print(f"  Mean Reward:    {metrics['mean_reward']:.4f}")
    print(f"  Std Reward:     {metrics['std_reward']:.4f}")
    print(f"  Median Reward:  {metrics['median_reward']:.4f}")
    print(f"  Min Reward:     {metrics['min_reward']:.4f}")
    print(f"  Max Reward:     {metrics['max_reward']:.4f}")
    print(f"  Q25 Reward:     {metrics['q25_reward']:.4f}")
    print(f"  Q75 Reward:     {metrics['q75_reward']:.4f}")
    print(f"  Num Samples:    {metrics['num_samples']}")
    print("="*60)


def print_sample_scores(predictions: List[Dict], scores: List[float], n: int = 3):
    """Print sample predictions with scores"""
    print(f"\nSample Predictions (Top {n}):")
    print("-"*60)
    
    # Get top N by score
    sorted_indices = np.argsort(scores)[::-1][:n]
    
    for rank, idx in enumerate(sorted_indices, 1):
        pred = predictions[idx]
        score = scores[idx]
        
        print(f"\nRank {rank} (Score: {score:.4f}):")
        print(f"  Question: {pred['question'][:100]}...")
        print(f"  Generated: {pred['generated_answer'][:150]}...")
        print(f"  Ground Truth: {pred['ground_truth'][:150]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO model predictions using reward model"
    )
    
    parser.add_argument(
        "--predictions_file",
        type=str,
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--reward_model_path",
        type=str,
        default="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        help="Path to reward model"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for evaluation"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PPO Model Evaluation")
    print("="*60)
    
    # Load reward model
    reward_model, tokenizer = load_reward_model(args.reward_model_path, args.device)
    
    # Load predictions
    predictions = load_predictions(args.predictions_file)
    
    # Compute reward scores
    scores = compute_reward_scores(
        reward_model,
        tokenizer,
        predictions,
        batch_size=args.batch_size,
    )
    
    # Compute metrics
    metrics = compute_metrics(scores)
    
    # Print results
    print_metrics(metrics)
    print_sample_scores(predictions, scores)
    
    # Save results
    save_evaluation_results(predictions, scores, metrics, args.output_file)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

