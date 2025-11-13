"""
Log training/validation responses to Weights & Biases during or after training.

This script can be run:
1. Standalone: Point to a checkpoint and log responses post-hoc
2. Integrated: Called from training script to log responses periodically

Usage:
    # Post-hoc logging after training
    python helpsteer3/log_responses_to_wandb.py \
        --model_path checkpoints/.../actor/huggingface \
        --data_file ~/data/helpsteer3_ppo/case1_question_only/validation.parquet \
        --wandb_run_id YOUR_RUN_ID \
        --num_samples 20

    # Or create new wandb run
    python helpsteer3/log_responses_to_wandb.py \
        --model_path checkpoints/.../actor/huggingface \
        --data_file ~/data/helpsteer3_ppo/case1_question_only/validation.parquet \
        --wandb_project helpsteer3_ppo_experiment \
        --wandb_name case1_validation_responses \
        --num_samples 20
"""

import argparse
import os
import random

import pandas as pd
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


def load_validation_data(data_file: str, num_samples: int = 20, seed: int = 42):
    """Load validation data from parquet file"""
    print(f"Loading validation data from {data_file}")
    df = pd.read_parquet(data_file)
    
    # Sample if needed
    if num_samples and len(df) > num_samples:
        df = df.sample(n=num_samples, random_state=seed)
    
    print(f"Loaded {len(df)} examples")
    return df


def generate_responses(model, tokenizer, prompts, max_new_tokens=768):
    """Generate responses for a batch of prompts"""
    responses = []
    
    for prompt in tqdm(prompts, desc="Generating responses"):
        # Apply chat template
        text = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        responses.append(response.strip())
    
    return responses


def log_to_wandb(
    df,
    responses,
    wandb_run,
    step=None,
    split="validation",
):
    """Log responses to wandb as a table"""
    
    # Create wandb table
    columns = ["index", "prompt_preview", "response", "response_length", "domain", "case"]
    table_data = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        # Get prompt preview (first 200 chars)
        prompt = row['prompt']
        if isinstance(prompt, list) and len(prompt) > 0:
            prompt_text = prompt[0].get('content', '')[:200] + "..."
        else:
            prompt_text = str(prompt)[:200] + "..."
        
        response = responses[idx]
        response_length = len(response.split())
        
        domain = row['extra_info'].get('domain', 'unknown') if 'extra_info' in row else 'unknown'
        case = row['extra_info'].get('case', 'unknown') if 'extra_info' in row else 'unknown'
        
        table_data.append([
            row['extra_info'].get('index', idx) if 'extra_info' in row else idx,
            prompt_text,
            response,
            response_length,
            domain,
            case
        ])
    
    # Create table
    table = wandb.Table(columns=columns, data=table_data)
    
    # Log table
    table_name = f"{split}_responses"
    if step is not None:
        table_name = f"{split}_responses_step_{step}"
    
    wandb_run.log({table_name: table}, step=step)
    
    print(f"✓ Logged {len(table_data)} responses to wandb as '{table_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Log model responses to Weights & Biases"
    )
    
    # Model args
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_file", type=str, required=True, help="Path to validation parquet file")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to log")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="Max tokens to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Wandb args
    parser.add_argument("--wandb_run_id", type=str, default=None, help="Resume existing wandb run")
    parser.add_argument("--wandb_project", type=str, default="helpsteer3_ppo_experiment", help="Wandb project")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--step", type=int, default=None, help="Training step number (for logging)")
    parser.add_argument("--split", type=str, default="validation", help="Split name (train/validation/test)")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Logging Responses to Weights & Biases")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_file}")
    print(f"Samples: {args.num_samples}")
    print(f"Project: {args.wandb_project}")
    print("="*80)
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"✓ Model loaded on {model.device}")
    
    # Load validation data
    df = load_validation_data(args.data_file, args.num_samples, args.seed)
    
    # Extract prompts
    prompts = df['prompt'].tolist()
    
    # Generate responses
    print(f"\nGenerating {len(prompts)} responses...")
    responses = generate_responses(model, tokenizer, prompts, args.max_new_tokens)
    
    # Initialize or resume wandb
    print("\nInitializing Weights & Biases...")
    if args.wandb_run_id:
        # Resume existing run
        wandb_run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="allow"
        )
        print(f"✓ Resumed wandb run: {args.wandb_run_id}")
    else:
        # Create new run
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f"{args.split}_responses",
            config={
                "model_path": args.model_path,
                "data_file": args.data_file,
                "num_samples": args.num_samples,
                "split": args.split,
                "step": args.step,
            }
        )
        print(f"✓ Created new wandb run: {wandb_run.id}")
    
    # Log to wandb
    print("\nLogging to Weights & Biases...")
    log_to_wandb(df, responses, wandb_run, step=args.step, split=args.split)
    
    # Log summary statistics
    response_lengths = [len(r.split()) for r in responses]
    wandb_run.log({
        f"{args.split}/response_length_mean": sum(response_lengths) / len(response_lengths),
        f"{args.split}/response_length_min": min(response_lengths),
        f"{args.split}/response_length_max": max(response_lengths),
        f"{args.split}/num_samples": len(responses),
    }, step=args.step)
    
    print("\n✓ Logging complete!")
    print(f"\nView in wandb: {wandb_run.url}")
    
    # Finish
    wandb_run.finish()


if __name__ == "__main__":
    main()

