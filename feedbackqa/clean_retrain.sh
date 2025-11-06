#!/bin/bash
# Clean retrain script - removes old model and trains fresh

set -e

echo "=========================================="
echo "Clean Retrain: Reward Model with BCE Loss"
echo "=========================================="
echo ""

# Step 1: Remove old model completely
echo "Step 1: Removing old model..."
if [ -d "./feedback_qa_reward_model" ]; then
    echo "  Deleting ./feedback_qa_reward_model/"
    rm -rf ./feedback_qa_reward_model
    echo "  ✓ Old model deleted"
else
    echo "  ℹ No existing model found (this is fine)"
fi
echo ""

# Step 2: Verify training script has correct config
echo "Step 2: Verifying training script..."
if grep -q "config.num_labels = 1" feedbackqa/rm_train.py; then
    echo "  ✓ Script has num_labels=1 (correct)"
else
    echo "  ✗ Script does NOT have num_labels=1!"
    echo "  Please check feedbackqa/rm_train.py"
    exit 1
fi

if grep -q "binary_cross_entropy_with_logits" feedbackqa/rm_train.py; then
    echo "  ✓ Script uses BCE loss (correct)"
else
    echo "  ✗ Script does NOT use BCE loss!"
    exit 1
fi
echo ""

# Step 3: Check data files exist
echo "Step 3: Checking data files..."
for file in "feedbackqa/feedback_train_rm.json" "feedbackqa/feedback_valid_rm.json" "feedbackqa/feedback_test_rm.json"; do
    if [ -f "$file" ]; then
        echo "  ✓ Found: $file"
    else
        echo "  ✗ Missing: $file"
        exit 1
    fi
done
echo ""

# Step 4: Start training
echo "Step 4: Starting training..."
echo "=========================================="
echo ""

python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 2e-5

# Step 5: Verify the trained model
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed! Now verifying..."
    echo "=========================================="
    echo ""
    
    # Check config.json
    if [ -f "./feedback_qa_reward_model/final_model/config.json" ]; then
        NUM_LABELS=$(grep -o '"num_labels": [0-9]*' ./feedback_qa_reward_model/final_model/config.json | grep -o '[0-9]*')
        if [ "$NUM_LABELS" = "1" ]; then
            echo "✓ Model config has num_labels=1 (correct!)"
        else
            echo "✗ Model config has num_labels=$NUM_LABELS (wrong!)"
            echo "Something went wrong during training."
            exit 1
        fi
    fi
    
    # Check training summary
    if [ -f "./feedback_qa_reward_model/training_summary.json" ]; then
        if grep -q "binary_cross_entropy" ./feedback_qa_reward_model/training_summary.json; then
            echo "✓ Training summary confirms BCE loss (correct!)"
        fi
    fi
    
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Model trained correctly!"
    echo "=========================================="
    echo ""
    echo "Run verification:"
    echo "  python feedbackqa/verify_bce_training.py"
    echo ""
else
    echo ""
    echo "❌ Training failed!"
    exit 1
fi

