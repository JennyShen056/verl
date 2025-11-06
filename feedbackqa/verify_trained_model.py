"""
Verify that your trained reward model is compatible with verl.

Usage:
    python feedbackqa/verify_trained_model.py --model_path ./feedback_qa_reward_model/final_model
"""

import argparse
import torch
from transformers import AutoModelForTokenClassification, AutoConfig, AutoTokenizer


def verify_model(model_path: str):
    """Verify that the model is verl-compatible"""
    print(f"Verifying model at: {model_path}\n")
    
    # Check 1: Load config
    print("✓ Loading model config...")
    config = AutoConfig.from_pretrained(model_path)
    
    # Check 2: Verify num_labels
    print(f"  num_labels: {config.num_labels}")
    assert config.num_labels == 1, f"Expected num_labels=1, got {config.num_labels}"
    print("  ✓ num_labels=1 (correct for verl)\n")
    
    # Check 3: Verify architecture
    print("✓ Checking architecture...")
    print(f"  architectures: {config.architectures}")
    arch_str = str(config.architectures)
    assert "TokenClassification" in arch_str, f"Expected TokenClassification, got {arch_str}"
    print("  ✓ Uses TokenClassification (correct for verl)\n")
    
    # Check 4: Load model
    print("✓ Loading model...")
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        print("  ✓ Model loaded successfully\n")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False
    
    # Check 5: Load tokenizer
    print("✓ Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"  Tokenizer loaded: {tokenizer.__class__.__name__}")
        print(f"  pad_token_id: {tokenizer.pad_token_id}")
        print(f"  eos_token_id: {tokenizer.eos_token_id}\n")
    except Exception as e:
        print(f"  ✗ Failed to load tokenizer: {e}")
        return False
    
    # Check 6: Test inference
    print("✓ Testing inference...")
    try:
        # Create sample input
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "The answer is 4."}
        ]
        
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        else:
            text = "Question: What is 2+2?\n\nAnswer: The answer is 4."
        
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Output shape: {logits.shape}")
        
        # Verify output shape
        expected_shape = (inputs['input_ids'].shape[0], inputs['input_ids'].shape[1], 1)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        print(f"  ✓ Output shape is correct: {logits.shape}\n")
        
        # Extract reward at EOS position (last token)
        reward_scores = logits[:, -1, 0]  # Take last position
        print(f"  Reward score at EOS: {reward_scores.item():.4f}")
        print(f"  ✓ Can extract reward score\n")
        
    except Exception as e:
        print(f"  ✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All checks passed
    print("="*60)
    print("✅ SUCCESS! Your model is verl-compatible!")
    print("="*60)
    print("\nYou can now use this model with verl PPO training:")
    print(f"  reward_model.enable=True \\")
    print(f"  reward_model.model.path={model_path}")
    print("\nSee feedbackqa/README_VERL_INTEGRATION.md for details.")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify verl-compatible reward model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./feedback_qa_reward_model/final_model",
        help="Path to the trained reward model"
    )
    args = parser.parse_args()
    
    try:
        verify_model(args.model_path)
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

