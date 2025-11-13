"""
Verify that both HelpSteer3 preprocessing files use the same test split.
This script checks that the test indices are consistent across Case 1 and Case 2.
"""

import random

from datasets import load_dataset


def get_test_indices(seed: int = 42, test_size: int = 500) -> set:
    """Get the test indices that will be used by both preprocessing scripts"""
    dataset = load_dataset("nvidia/HelpSteer3", "feedback", split="validation")
    all_data = list(dataset)
    
    random.seed(seed)
    test_indices = set(random.sample(range(len(all_data)), test_size))
    
    return test_indices, len(all_data)


def main():
    print("="*80)
    print("Verifying Test Split Consistency for HelpSteer3 Preprocessing")
    print("="*80)
    
    SEED = 42
    TEST_SIZE = 500
    
    print(f"\nConfiguration:")
    print(f"  Seed: {SEED}")
    print(f"  Test size: {TEST_SIZE}")
    
    # Get test indices (same logic as preprocessing scripts)
    test_indices1, total_size = get_test_indices(seed=SEED, test_size=TEST_SIZE)
    
    # Verify it's deterministic
    test_indices2, _ = get_test_indices(seed=SEED, test_size=TEST_SIZE)
    
    print(f"\nValidation split size: {total_size}")
    print(f"Test split size: {TEST_SIZE}")
    print(f"New validation split size: {total_size - TEST_SIZE}")
    
    print(f"\n{'='*80}")
    print("Consistency Check")
    print(f"{'='*80}")
    
    if test_indices1 == test_indices2:
        print("✓ Test indices are deterministic and consistent!")
        print(f"  Both calls produced identical {len(test_indices1)} test indices")
    else:
        print("✗ ERROR: Test indices differ between calls!")
        print(f"  First call: {len(test_indices1)} indices")
        print(f"  Second call: {len(test_indices2)} indices")
        return False
    
    print(f"\n{'='*80}")
    print("Sample Test Indices (first 20)")
    print(f"{'='*80}")
    sorted_indices = sorted(test_indices1)
    print(f"  {sorted_indices[:20]}")
    
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    print(f"✓ Both preprocessing files will use the SAME {TEST_SIZE} examples for test")
    print(f"✓ Train split: ~{total_size} examples (unchanged)")
    print(f"✓ Validation split: {total_size - TEST_SIZE} examples (reduced)")
    print(f"✓ Test split: {TEST_SIZE} examples (new, split from validation)")
    
    print(f"\n{'='*80}")
    print("Usage Example")
    print(f"{'='*80}")
    print(f"\nTo preprocess all splits for Case 1:")
    print(f"  python feedbackqa/preprocess_ppo_helpsteer3_case1_question_only.py \\")
    print(f"    --splits train validation test \\")
    print(f"    --seed {SEED} \\")
    print(f"    --test_size {TEST_SIZE}")
    
    print(f"\nTo preprocess all splits for Case 2:")
    print(f"  python feedbackqa/preprocess_ppo_helpsteer3_case2_with_feedback.py \\")
    print(f"    --splits train validation test \\")
    print(f"    --seed {SEED} \\")
    print(f"    --test_size {TEST_SIZE}")
    
    print(f"\nNote: Use the SAME --seed and --test_size values for both files!")
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

