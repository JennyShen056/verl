"""
Inference script for HelpSteer3 trained models.
Generates responses for test set using a trained model.
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_test_data(test_size: int = 500, seed: int = 42) -> List[Dict]:
    """Load HelpSteer3 test split"""
    print(f"Loading HelpSteer3 test split (from validation)...")
    dataset = load_dataset("nvidia/HelpSteer3", "feedback", split="validation")
    all_data = list(dataset)
    
    # Use same logic as preprocessing to get test split
    import random
    random.seed(seed)
    test_indices = set(random.sample(range(len(all_data)), test_size))
    test_data = [all_data[i] for i in sorted(test_indices)]
    
    print(f"Loaded {len(test_data)} test examples")
    return test_data


def generate_responses(
    model,
    tokenizer,
    test_data: List[Dict],
    batch_size: int = 8,
    max_new_tokens: int = 768,
) -> List[Dict]:
    """Generate responses for test data"""
    
    results = []
    
    for i in tqdm(range(0, len(test_data), batch_size), desc="Generating responses"):
        batch = test_data[i:i + batch_size]
        
        # Prepare batch of conversations
        conversations = [example["context"] for example in batch]
        
        # Apply chat template
        texts = [
            tokenizer.apply_chat_template(
                conv,
                tokenize=False,
                add_generation_prompt=True
            )
            for conv in conversations
        ]
        
        # Tokenize
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
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
        
        # Decode responses
        for j, example in enumerate(batch):
            # Get only the generated part (exclude prompt)
            generated_ids = outputs[j][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            results.append({
                "index": i + j,
                "context": example["context"],
                "domain": example.get("domain", ""),
                "language": example.get("language", ""),
                "generated_response": response.strip(),
                "reference_response1": example.get("response1", ""),
                "reference_response2": example.get("response2", ""),
                "reference_feedback1": example.get("feedback1", []),
                "reference_feedback2": example.get("feedback2", []),
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference on HelpSteer3 test set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output_file", type=str, required=True, help="Output JSON file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=768, help="Max tokens to generate")
    parser.add_argument("--test_size", type=int, default=500, help="Test set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for test split")
    
    args = parser.parse_args()
    
    print("="*80)
    print("HelpSteer3 Inference")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print("="*80)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    
    # Load test data
    test_data = load_test_data(test_size=args.test_size, seed=args.seed)
    
    # Generate responses
    print(f"\nGenerating responses for {len(test_data)} examples...")
    results = generate_responses(
        model,
        tokenizer,
        test_data,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved {len(results)} predictions to {args.output_file}")


if __name__ == "__main__":
    main()

