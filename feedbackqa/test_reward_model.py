#!/usr/bin/env python3
"""
Quick test script to verify reward model can be loaded and used.
This checks if the model checkpoint is valid and can generate reward scores.
"""

import argparse
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer


def test_reward_model(model_path: str, input_tokenizer_path: str = None):
    """Test if the reward model can be loaded and run inference."""
    
    print("=" * 80)
    print("Reward Model Test")
    print("=" * 80)
    print(f"Model path: {model_path}")
    print(f"Input tokenizer path: {input_tokenizer_path or 'Same as model'}")
    print()
    
    # Step 1: Check if path exists
    print("Step 1: Checking if model path exists...")
    if not os.path.exists(model_path):
        print(f"❌ ERROR: Model path not found: {model_path}")
        return False
    print("✅ Model path exists")
    print()
    
    # Step 2: Check for required files
    print("Step 2: Checking for required files...")
    required_files = []
    optional_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
    
    found_files = []
    for file in optional_files:
        file_path = os.path.join(model_path, file)
        if os.path.exists(file_path):
            found_files.append(file)
            print(f"  ✅ Found: {file}")
    
    if not found_files:
        print("  ❌ ERROR: No model files found!")
        print("  Expected at least one of: pytorch_model.bin, model.safetensors")
        return False
    print()
    
    # Step 3: Try loading config
    print("Step 3: Loading model config...")
    try:
        config = AutoConfig.from_pretrained(model_path)
        print(f"✅ Config loaded successfully")
        print(f"  Model type: {getattr(config, 'model_type', 'Unknown')}")
        print(f"  Num labels: {getattr(config, 'num_labels', 'Unknown')}")
    except Exception as e:
        print(f"❌ ERROR loading config: {e}")
        print()
        print("💡 This might be a checkpoint-only directory.")
        print("   The model can still work if you specify input_tokenizer")
        config = None
    print()
    
    # Step 4: Try loading tokenizer
    print("Step 4: Loading tokenizer...")
    tokenizer_path = input_tokenizer_path or model_path
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✅ Tokenizer loaded from: {tokenizer_path}")
        print(f"  Vocab size: {len(tokenizer)}")
        print(f"  PAD token: {tokenizer.pad_token}")
        print(f"  EOS token: {tokenizer.eos_token}")
    except Exception as e:
        print(f"❌ ERROR loading tokenizer: {e}")
        if not input_tokenizer_path:
            print()
            print("💡 Try specifying --input-tokenizer (e.g., meta-llama/Llama-3.1-8B-Instruct)")
        return False
    print()
    
    # Step 5: Try loading model
    print("Step 5: Loading model...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu"  # Use CPU for testing
        )
        print("✅ Model loaded successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")
        return False
    print()
    
    # Step 6: Try a test inference
    print("Step 6: Testing inference...")
    try:
        test_text = "Question: What is 2+2?\nAnswer: The answer is 4."
        inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get score at last token
        last_token_idx = inputs['attention_mask'].sum(dim=1) - 1
        score = logits[0, last_token_idx, :].item()
        
        print("✅ Inference successful!")
        print(f"  Input length: {inputs['input_ids'].shape[1]} tokens")
        print(f"  Output shape: {logits.shape}")
        print(f"  Sample score: {score:.4f}")
    except Exception as e:
        print(f"❌ ERROR during inference: {e}")
        import traceback
        traceback.print_exc()
        return False
    print()
    
    # Success!
    print("=" * 80)
    print("✅ SUCCESS! Reward model is ready to use")
    print("=" * 80)
    print()
    print("Configuration for PPO training:")
    print(f'  reward_model.model.path="{model_path}"')
    if input_tokenizer_path:
        print(f'  reward_model.model.input_tokenizer="{input_tokenizer_path}"')
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test if reward model can be loaded and used for PPO training"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/workspace/verl/feedbackqa/feedback_qa_reward_model",
        help="Path to reward model checkpoint"
    )
    parser.add_argument(
        "--input-tokenizer",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to tokenizer (if different from model path)"
    )
    
    args = parser.parse_args()
    
    success = test_reward_model(args.model_path, args.input_tokenizer)
    
    if not success:
        print()
        print("=" * 80)
        print("❌ FAILED: Reward model cannot be used")
        print("=" * 80)
        print()
        print("Troubleshooting:")
        print("1. Check if model training completed successfully")
        print("2. Verify model path is correct")
        print("3. Try specifying --input-tokenizer explicitly")
        print("4. Check GPU memory if loading fails")
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

