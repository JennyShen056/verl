"""
Verify that your trained reward model is compatible with verl.

Usage:
    python feedbackqa/verify_trained_model.py --model_path ./feedback_qa_reward_model/final_model
"""

import argparse

import torch
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)


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
    
    # Check 3: Verify problem type
    print("✓ Checking problem type...")
    print(f"  problem_type: {config.problem_type}")
    if hasattr(config, 'problem_type'):
        assert config.problem_type == "regression", f"Expected regression, got {config.problem_type}"
        print("  ✓ Uses regression (correct for verl)\n")
    else:
        print("  ⚠ problem_type not set in config (okay if num_labels=1)\n")
    
    # Check 4: Load model
    print("✓ Loading model...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        print("  ✓ Model loaded successfully")
        print(f"  Model type: {model.__class__.__name__}")
        print(f"  Classifier output features: {model.classifier.out_features}\n")
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
        
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Output logits shape: {logits.shape}")
        
        # For num_labels=1, output should be [batch_size, 1]
        batch_size = inputs['input_ids'].shape[0]
        expected_shape = (batch_size, 1)
        assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
        print(f"  ✓ Output shape is correct: {logits.shape}")
        print(f"    (batch_size={batch_size}, num_labels=1)\n")
        
        # Extract scalar reward score
        reward_score = logits[0, 0].item()  # First batch item, single output
        print(f"  Reward score: {reward_score:.4f}")
        print(f"  ✓ Can extract scalar reward score\n")
        
        # Test with a batch
        print("✓ Testing batch inference...")
        texts = [
            tokenizer.apply_chat_template([
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "The answer is 4."}
            ], tokenize=False, add_generation_prompt=False) if hasattr(tokenizer, 'apply_chat_template') 
            else "Question: What is 2+2?\n\nAnswer: The answer is 4.",
            
            tokenizer.apply_chat_template([
                {"role": "user", "content": "What color is the sky?"},
                {"role": "assistant", "content": "The sky is blue."}
            ], tokenize=False, add_generation_prompt=False) if hasattr(tokenizer, 'apply_chat_template')
            else "Question: What color is the sky?\n\nAnswer: The sky is blue."
        ]
        
        batch_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            batch_outputs = model(**batch_inputs)
        
        batch_logits = batch_outputs.logits
        print(f"  Batch input shape: {batch_inputs['input_ids'].shape}")
        print(f"  Batch output shape: {batch_logits.shape}")
        
        batch_size = batch_inputs['input_ids'].shape[0]
        expected_batch_shape = (batch_size, 1)
        assert batch_logits.shape == expected_batch_shape, f"Expected {expected_batch_shape}, got {batch_logits.shape}"
        
        batch_scores = batch_logits.squeeze(-1).cpu().numpy()
        print(f"  Batch reward scores: {batch_scores}")
        print(f"  ✓ Batch inference works correctly\n")
        
    except Exception as e:
        print(f"  ✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # All checks passed
    print("="*70)
    print("✅ SUCCESS! Your model is verl-compatible!")
    print("="*70)
    print("\n📋 Model Summary:")
    print(f"  • Architecture: AutoModelForSequenceClassification")
    print(f"  • num_labels: 1 (scalar regression)")
    print(f"  • problem_type: {config.problem_type if hasattr(config, 'problem_type') else 'not set'}")
    print(f"  • Output shape: [batch_size, 1]")
    print(f"  • Output range: unbounded (-∞, +∞)")
    print("\n🚀 You can now use this model with verl PPO training:")
    print(f"  reward_model.enable=True \\")
    print(f"  reward_model.model.path={model_path}")
    print("\n📚 See feedbackqa/REWARD_MODEL_FORMAT_EXPLANATION.md for details.")
    print("="*70)
    
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

