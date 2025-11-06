#!/bin/bash
# Complete Evaluation Pipeline
# =============================
# Merges checkpoints, runs inference, evaluates, and compares results

set -e

echo "========================================="
echo "Complete Evaluation Pipeline"
echo "========================================="
echo "This script will:"
echo "  1. Merge Case 1 checkpoint to HuggingFace format"
echo "  2. Merge Case 2 checkpoint to HuggingFace format"
echo "  3. Run inference on test set for both cases"
echo "  4. Evaluate using reward model"
echo "  5. Compare results statistically"
echo "========================================="
echo ""

# Configuration
CASE1_NAME="case1_question_only_baseline"
CASE2_NAME="case2_with_feedback_experimental"
GLOBAL_STEP=10  # Default to step 1, or use first argument
OUTPUT_DIR="outputs"

echo "Configuration:"
echo "  Case 1: $CASE1_NAME"
echo "  Case 2: $CASE2_NAME"
echo "  Global Step: $GLOBAL_STEP"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Merge Case 1 checkpoint
echo "Step 1/5: Merging Case 1 checkpoint..."
echo "======================================="
# bash feedbackqa/merge_checkpoint.sh "$CASE1_NAME" "$GLOBAL_STEP"
bash feedbackqa/merge_checkpoint.sh "$CASE1_NAME" "$GLOBAL_STEP"
echo ""

# Step 2: Merge Case 2 checkpoint
# echo "Step 2/5: Merging Case 2 checkpoint..."
# echo "======================================="
# bash feedbackqa/merge_checkpoint.sh "$CASE2_NAME" "$GLOBAL_STEP"
# echo ""

# Paths to merged models
CASE1_MODEL="checkpoints/feedback_qa_experiment/${CASE1_NAME}/global_step_${GLOBAL_STEP}/actor/huggingface"
# CASE2_MODEL="checkpoints/feedback_qa_experiment/${CASE2_NAME}/global_step_${GLOBAL_STEP}/actor/huggingface"

# Step 3: Run inference on Case 1
echo "Step 3/5: Running inference for Case 1..."
echo "=========================================="
python feedbackqa/inference.py \
    --model_path "$CASE1_MODEL" \
    --test_file feedbackqa/feedback_test_ppo.json \
    --output_file "$OUTPUT_DIR/case1_predictions.json" \
    --batch_size 8 \
    --max_new_tokens 512
echo ""

# Step 4: Run inference on Case 2
# echo "Step 4/5: Running inference for Case 2..."
# echo "=========================================="
# python feedbackqa/inference.py \
#     --model_path "$CASE2_MODEL" \
#     --test_file feedbackqa/feedback_test_ppo.json \
#     --output_file "$OUTPUT_DIR/case2_predictions.json" \
#     --batch_size 8 \
#     --max_new_tokens 512
# echo ""

# Step 5: Evaluate Case 1
echo "Step 5/5: Evaluating Case 1..."
echo "==============================="
python feedbackqa/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case1_predictions.json" \
    --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --output_file "$OUTPUT_DIR/case1_evaluation.json" \
    --batch_size 8
echo ""

# Step 6: Evaluate Case 2
# echo "Step 6/6: Evaluating Case 2..."
# echo "==============================="
# python feedbackqa/evaluate.py \
#     --predictions_file "$OUTPUT_DIR/case2_predictions.json" \
#     --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
#     --output_file "$OUTPUT_DIR/case2_evaluation.json" \
#     --batch_size 8
# echo ""

# Step 7: Compare results
# echo "Step 7/7: Comparing results..."
# echo "==============================="
# python feedbackqa/compare_results.py \
#     --case1_file "$OUTPUT_DIR/case1_evaluation.json" \
#     --case2_file "$OUTPUT_DIR/case2_evaluation.json" \
#     --output_file "$OUTPUT_DIR/comparison_report.json"
# echo ""

# echo "========================================="
# echo "Evaluation Pipeline Complete!"
# echo "========================================="
# echo ""
# echo "Results saved to:"
# echo "  $OUTPUT_DIR/case1_predictions.json"
# echo "  $OUTPUT_DIR/case1_evaluation.json"
# echo "  $OUTPUT_DIR/case2_predictions.json"
# echo "  $OUTPUT_DIR/case2_evaluation.json"
# echo "  $OUTPUT_DIR/comparison_report.json"
# echo ""
# echo "Next steps:"
# echo "  1. Review comparison_report.json for statistical results"
# echo "  2. Check individual evaluation files for detailed scores"
# echo "  3. Inspect predictions to understand model behavior"

