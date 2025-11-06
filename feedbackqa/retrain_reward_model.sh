#!/bin/bash
# Quick script to retrain the reward model with verl-compatible configuration

set -e  # Exit on error

echo "=========================================="
echo "Retraining Reward Model (verl-compatible)"
echo "=========================================="
echo ""

# Check if old model exists
if [ -d "./feedback_qa_reward_model" ]; then
    echo "⚠️  Old reward model found at ./feedback_qa_reward_model"
    read -p "Delete old model and retrain? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Deleting old model..."
        rm -rf ./feedback_qa_reward_model
        echo "✓ Old model deleted"
    else
        echo "❌ Aborted. Please delete old model manually or choose a different output directory."
        exit 1
    fi
fi

# Check if data files exist
echo ""
echo "Checking data files..."
for file in "feedbackqa/feedback_train_rm.json" "feedbackqa/feedback_valid_rm.json" "feedbackqa/feedback_test_rm.json"; do
    if [ ! -f "$file" ]; then
        echo "❌ Error: $file not found!"
        echo "Please ensure data files are in the feedbackqa/ directory."
        exit 1
    fi
    echo "✓ Found: $file"
done

# Set default parameters (can override with environment variables)
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

echo ""
echo "Training Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Batch size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max length: $MAX_LENGTH"
echo ""
echo "Starting training..."
echo "=========================================="
echo ""

# Run training
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --max_length "$MAX_LENGTH"

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Training completed successfully!"
    echo "=========================================="
    echo ""
    
    # Verify the model
    echo "Verifying model compatibility..."
    echo ""
    
    if python feedbackqa/verify_trained_model.py --model_path ./feedback_qa_reward_model/final_model; then
        echo ""
        echo "=========================================="
        echo "✅ SUCCESS! Model is verl-compatible!"
        echo "=========================================="
        echo ""
        echo "Next steps:"
        echo "  1. Preprocess PPO data:"
        echo "     python feedbackqa/preprocess_ppo_case1_question_only.py"
        echo "     python feedbackqa/preprocess_ppo_case2_with_feedback.py"
        echo ""
        echo "  2. Run PPO training:"
        echo "     bash feedbackqa/run_ppo_case1_question_only.sh"
        echo "     bash feedbackqa/run_ppo_case2_with_feedback.sh"
        echo ""
    else
        echo ""
        echo "⚠️  Warning: Model verification failed!"
        echo "Please check the error messages above."
        echo "See FIX_REWARD_MODEL_ISSUE.md for troubleshooting."
        exit 1
    fi
else
    echo ""
    echo "❌ Training failed!"
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "  - Out of memory: Reduce --batch_size"
    echo "  - Missing data files: Check file paths"
    echo "  - Model download issues: Check internet connection"
    exit 1
fi

