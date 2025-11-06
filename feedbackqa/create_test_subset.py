#!/usr/bin/env python3
"""
Create Test Subset
==================

Randomly select N samples from test set for faster evaluation.

Usage:
    python feedbackqa/create_test_subset.py \
        --input_file feedbackqa/feedback_test_ppo.json \
        --output_file feedbackqa/feedback_test_subset_100.json \
        --num_samples 100 \
        --seed 42
"""

import argparse
import json
import random


def create_subset(input_file: str, output_file: str, num_samples: int = 100, seed: int = 42):
    """Create a random subset of test data"""
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load full test set
    print(f"Loading test data from: {input_file}")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    print(f"  Total samples: {len(data)}")
    
    # Randomly select subset
    if num_samples >= len(data):
        print(f"  Requested {num_samples} samples, but only {len(data)} available")
        print(f"  Using all {len(data)} samples")
        subset = data
    else:
        subset = random.sample(data, num_samples)
        print(f"  Selected {num_samples} random samples (seed={seed})")
    
    # Save subset
    with open(output_file, 'w') as f:
        json.dump(subset, f, indent=2)
    
    print(f"\n✓ Subset saved to: {output_file}")
    print(f"  Samples: {len(subset)}")
    
    # Print sample info
    if subset:
        print(f"\nSample question:")
        print(f"  {subset[0]['question'][:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Create random subset of test data"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to full test data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save subset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to select"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Creating Test Subset")
    print("="*60)
    
    create_subset(
        args.input_file,
        args.output_file,
        args.num_samples,
        args.seed
    )
    
    print("\n" + "="*60)
    print("Subset Creation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

