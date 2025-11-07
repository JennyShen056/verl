#!/bin/bash
# Complete Evaluation Pipeline
# =============================
# Merges checkpoints, runs inference, evaluates, and compares results
#
# Usage:
#   bash feedbackqa/run_full_evaluation.sh
#
# Note: Edit CASE1_GLOBAL_STEP and CASE2_GLOBAL_STEP below to evaluate different checkpoints

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
CASE1_GLOBAL_STEP=21  # Global step for Case 1
CASE2_GLOBAL_STEP=41  # Global step for Case 2
OUTPUT_DIR="outputs"

echo "Configuration:"
echo "  Case 1: $CASE1_NAME (step $CASE1_GLOBAL_STEP)"
echo "  Case 2: $CASE2_NAME (step $CASE2_GLOBAL_STEP)"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 0: Create test subset (100 samples) for faster evaluation
echo "Step 0/7: Creating test subset (100 samples)..."
echo "================================================"
python feedbackqa/create_test_subset.py \
    --input_file feedbackqa/feedback_test_ppo.json \
    --output_file feedbackqa/feedback_test_subset_100.json \
    --num_samples 100 \
    --seed 42
echo ""

# Use subset for evaluation
TEST_FILE="feedbackqa/feedback_test_subset_100.json"

# Step 1: Merge Case 1 checkpoint
echo "Step 1/7: Merging Case 1 checkpoint..."
echo "======================================="
bash feedbackqa/merge_checkpoint.sh "$CASE1_NAME" "$CASE1_GLOBAL_STEP"
echo ""

# Step 2: Merge Case 2 checkpoint
echo "Step 2/7: Merging Case 2 checkpoint..."
echo "======================================="
bash feedbackqa/merge_checkpoint.sh "$CASE2_NAME" "$CASE2_GLOBAL_STEP"
echo ""

# Paths to merged models
CASE1_MODEL="checkpoints/feedback_qa_experiment/${CASE1_NAME}/global_step_${CASE1_GLOBAL_STEP}/actor/huggingface"
CASE2_MODEL="checkpoints/feedback_qa_experiment/${CASE2_NAME}/global_step_${CASE2_GLOBAL_STEP}/actor/huggingface"

# Step 3: Run inference on Case 1
echo "Step 3/7: Running inference for Case 1..."
echo "=========================================="
echo "Using test file: $TEST_FILE (100 samples)"
python feedbackqa/inference.py \
    --model_path "$CASE1_MODEL" \
    --test_file "$TEST_FILE" \
    --output_file "$OUTPUT_DIR/case1_predictions.json" \
    --batch_size 8 \
    --max_new_tokens 512
echo ""

# Step 4: Run inference on Case 2
echo "Step 4/7: Running inference for Case 2..."
echo "=========================================="
echo "Using test file: $TEST_FILE (100 samples)"
python feedbackqa/inference.py \
    --model_path "$CASE2_MODEL" \
    --test_file "$TEST_FILE" \
    --output_file "$OUTPUT_DIR/case2_predictions.json" \
    --batch_size 8 \
    --max_new_tokens 512
echo ""

# Step 5: Evaluate Case 1
echo "Step 5/7: Evaluating Case 1..."
echo "==============================="
python feedbackqa/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case1_predictions.json" \
    --reward_model_path Jennny/qa_rm \
    --output_file "$OUTPUT_DIR/case1_evaluation.json" \
    --batch_size 8
echo ""

# Step 6: Evaluate Case 2
echo "Step 6/7: Evaluating Case 2..."
echo "==============================="
python feedbackqa/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case2_predictions.json" \
    --reward_model_path Jennny/qa_rm \
    --output_file "$OUTPUT_DIR/case2_evaluation.json" \
    --batch_size 8
echo ""

# Step 7: Compare results
echo "Step 7/7: Comparing results..."
echo "==============================="
python feedbackqa/compare_results.py \
    --case1_file "$OUTPUT_DIR/case1_evaluation.json" \
    --case2_file "$OUTPUT_DIR/case2_evaluation.json" \
    --output_file "$OUTPUT_DIR/comparison_report.json"
echo ""

echo "========================================="
echo "Evaluation Pipeline Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  $OUTPUT_DIR/case1_predictions.json (100 samples)"
echo "  $OUTPUT_DIR/case1_evaluation.json"
echo "  $OUTPUT_DIR/case2_predictions.json (100 samples)"
echo "  $OUTPUT_DIR/case2_evaluation.json"
echo "  $OUTPUT_DIR/comparison_report.json"
echo ""
echo "Test subset saved to:"
echo "  feedbackqa/feedback_test_subset_100.json (same 100 samples for both cases)"
echo ""
echo "Next steps:"
echo "  1. Review comparison_report.json for statistical results"
echo "  2. Check individual evaluation files for detailed scores"
echo "  3. Inspect predictions to understand model behavior"

