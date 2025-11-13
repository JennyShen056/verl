"""
Verify that HelpSteer3 setup is correct before running training.
Checks data preprocessing, file existence, and configuration.
"""

import os
import sys


def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description}: {filepath} NOT FOUND")
        return False


def check_dir_exists(dirpath, description):
    """Check if a directory exists"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✓ {description}: {dirpath}")
        return True
    else:
        print(f"✗ {description}: {dirpath} NOT FOUND")
        return False


def main():
    print("="*80)
    print("HelpSteer3 Setup Verification")
    print("="*80)
    
    all_checks_passed = True
    
    # Check preprocessing scripts
    print("\n1. Checking preprocessing scripts...")
    print("-"*80)
    all_checks_passed &= check_file_exists(
        "helpsteer3/preprocess_ppo_helpsteer3_case1_question_only.py",
        "Case 1 preprocessing script"
    )
    all_checks_passed &= check_file_exists(
        "helpsteer3/preprocess_ppo_helpsteer3_case2_with_feedback.py",
        "Case 2 preprocessing script"
    )
    
    # Check training scripts
    print("\n2. Checking training scripts...")
    print("-"*80)
    all_checks_passed &= check_file_exists(
        "helpsteer3/run_ppo_case1_question_only.sh",
        "Case 1 training script"
    )
    all_checks_passed &= check_file_exists(
        "helpsteer3/run_ppo_case2_with_feedback.sh",
        "Case 2 training script"
    )
    
    # Check evaluation scripts
    print("\n3. Checking evaluation scripts...")
    print("-"*80)
    all_checks_passed &= check_file_exists(
        "helpsteer3/inference.py",
        "Inference script"
    )
    all_checks_passed &= check_file_exists(
        "helpsteer3/evaluate.py",
        "Evaluation script"
    )
    all_checks_passed &= check_file_exists(
        "helpsteer3/compare_results.py",
        "Comparison script"
    )
    all_checks_passed &= check_file_exists(
        "helpsteer3/run_full_evaluation.sh",
        "Full evaluation pipeline"
    )
    
    # Check data directories
    print("\n4. Checking data directories...")
    print("-"*80)
    home = os.path.expanduser("~")
    
    case1_dir = os.path.join(home, "data/helpsteer3_ppo/case1_question_only")
    case2_dir = os.path.join(home, "data/helpsteer3_ppo/case2_with_feedback")
    
    case1_exists = check_dir_exists(case1_dir, "Case 1 data directory")
    case2_exists = check_dir_exists(case2_dir, "Case 2 data directory")
    
    # Check for preprocessed data files
    if case1_exists:
        print("\n  Checking Case 1 parquet files:")
        for split in ["train", "validation", "test"]:
            filepath = os.path.join(case1_dir, f"{split}.parquet")
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"    ✓ {split}.parquet ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ {split}.parquet NOT FOUND")
                all_checks_passed = False
    else:
        print("  ⚠️  Run preprocessing first: python helpsteer3/preprocess_ppo_helpsteer3_case1_question_only.py --splits train validation test")
        all_checks_passed = False
    
    if case2_exists:
        print("\n  Checking Case 2 parquet files:")
        for split in ["train", "validation", "test"]:
            filepath = os.path.join(case2_dir, f"{split}.parquet")
            if os.path.exists(filepath):
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"    ✓ {split}.parquet ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ {split}.parquet NOT FOUND")
                all_checks_passed = False
    else:
        print("  ⚠️  Run preprocessing first: python helpsteer3/preprocess_ppo_helpsteer3_case2_with_feedback.py --splits train validation test")
        all_checks_passed = False
    
    # Check Python dependencies
    print("\n5. Checking Python dependencies...")
    print("-"*80)
    
    try:
        import datasets
        print(f"✓ datasets ({datasets.__version__})")
    except ImportError:
        print("✗ datasets not installed")
        all_checks_passed = False
    
    try:
        import pandas
        print(f"✓ pandas ({pandas.__version__})")
    except ImportError:
        print("✗ pandas not installed")
        all_checks_passed = False
    
    try:
        import transformers
        print(f"✓ transformers ({transformers.__version__})")
    except ImportError:
        print("✗ transformers not installed")
        all_checks_passed = False
    
    try:
        import torch
        print(f"✓ torch ({torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        print("✗ torch not installed")
        all_checks_passed = False
    
    try:
        import scipy
        print(f"✓ scipy ({scipy.__version__})")
    except ImportError:
        print("✗ scipy not installed (needed for evaluation)")
        all_checks_passed = False
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all_checks_passed:
        print("\n✓ All checks passed! You're ready to run training.")
        print("\nNext steps:")
        print("  1. bash helpsteer3/run_ppo_case1_question_only.sh")
        print("  2. bash helpsteer3/run_ppo_case2_with_feedback.sh")
        print("  3. bash helpsteer3/run_full_evaluation.sh")
        return 0
    else:
        print("\n✗ Some checks failed. Please address the issues above.")
        print("\nTo preprocess data:")
        print("  python helpsteer3/preprocess_ppo_helpsteer3_case1_question_only.py --splits train validation test")
        print("  python helpsteer3/preprocess_ppo_helpsteer3_case2_with_feedback.py --splits train validation test")
        return 1


if __name__ == "__main__":
    sys.exit(main())

