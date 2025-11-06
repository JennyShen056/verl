#!/usr/bin/env python3
"""
Inference Script for Trained PPO Models
========================================

Generate answers on test set using trained Case 1 or Case 2 models.

Usage:
    python feedbackqa/inference.py \
        --model_path checkpoints/feedback_qa_experiment/case1_question_only_baseline/global_step_1/actor/huggingface \
        --test_file feedbackqa/feedback_test_ppo.json \
        --output_file outputs/case1_predictions.json \
        --batch_size 8 \
        --max_new_tokens 512
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    """Load trained model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    
    model.eval()
    
    print(f"✓ Model loaded: {model.__class__.__name__}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"  Device: {next(model.parameters()).device}")
    
    return model, tokenizer


def load_test_data(test_file: str) -> List[Dict]:
    """Load test data"""
    print(f"Loading test data from: {test_file}")
    with open(test_file, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} test examples")
    return data


def format_prompt(question: str, tokenizer) -> str:
    """Format question as prompt using chat template"""
    messages = [
        {"role": "user", "content": question}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Fallback format
        prompt = f"Question: {question}\n\nAnswer:"
    
    return prompt


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[str]:
    """Generate responses for a batch of prompts"""
    
    # Tokenize
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part (excluding input prompt)
    prompt_lengths = inputs['input_ids'].shape[1]
    generated_ids = outputs[:, prompt_lengths:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    return responses


def run_inference(
    model,
    tokenizer,
    test_data: List[Dict],
    batch_size: int = 8,
    max_new_tokens: int = 512,
) -> List[Dict]:
    """Run inference on test data"""
    
    print(f"\nRunning inference...")
    print(f"  Batch size: {batch_size}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Total examples: {len(test_data)}")
    
    results = []
    
    # Process in batches
    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating"):
        batch = test_data[i:i+batch_size]
        
        # Format prompts
        prompts = [format_prompt(ex["question"], tokenizer) for ex in batch]
        
        # Generate
        responses = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
        )
        
        # Collect results
        for example, response in zip(batch, responses):
            results.append({
                "question": example["question"],
                "ground_truth": example["section_content"],
                "feedback": example.get("feedback", ""),
                "rating": example.get("rating", ""),
                "generated_answer": response,
            })
    
    return results


def save_results(results: List[Dict], output_file: str):
    """Save inference results"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"  Total predictions: {len(results)}")
    
    # Print sample
    print(f"\nSample prediction:")
    sample = results[0]
    print(f"  Question: {sample['question'][:100]}...")
    print(f"  Generated: {sample['generated_answer'][:200]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained PPO model"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged HuggingFace model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="feedbackqa/feedback_test_ppo.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (auto, cuda, cpu)"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PPO Model Inference")
    print("="*60)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)
    
    # Load test data
    test_data = load_test_data(args.test_file)
    
    # Run inference
    results = run_inference(
        model,
        tokenizer,
        test_data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Save results
    save_results(results, args.output_file)
    
    print("\n" + "="*60)
    print("Inference Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

