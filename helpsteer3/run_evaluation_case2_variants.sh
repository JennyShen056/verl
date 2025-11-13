#!/bin/bash
# Complete Evaluation Pipeline for HelpSteer3 Case 2 Variants
# ============================================================
# Compares two Case 2 variants:
#   - Case 2 (All Feedback): Uses all feedback items from the list
#   - Case 2 (Single Feedback): Uses one randomly selected feedback item
#
# Usage:
#   bash helpsteer3/run_evaluation_case2_variants.sh
#
# Note: Edit CASE2_ALL_GLOBAL_STEP and CASE2_SINGLE_GLOBAL_STEP below to evaluate different checkpoints

set -e

echo "========================================="
echo "HelpSteer3 Case 2 Variants Evaluation"
echo "========================================="
echo "This script will:"
echo "  1. Merge Case 2 (All Feedback) checkpoint to HuggingFace format"
echo "  2. Merge Case 2 (Single Feedback) checkpoint to HuggingFace format"
echo "  3. Run inference on test set for both variants"
echo "  4. Evaluate using reward model (Jennny/llama3_help_rm)"
echo "  5. Compare results statistically"
echo "========================================="
echo ""

# Configuration
CASE2_ALL_NAME="helpsteer3_case2_with_feedback_experimental"
CASE2_SINGLE_NAME="helpsteer3_case2_single_feedback_variant"
CASE2_ALL_GLOBAL_STEP=201  # Global step for Case 2 (All Feedback)
CASE2_SINGLE_GLOBAL_STEP=209  # Global step for Case 2 (Single Feedback)
OUTPUT_DIR="helpsteer3/outputs/case2_variants_comparison"
TEST_SIZE=500  # Number of test samples (same as preprocessing)
SEED=42  # Same seed as preprocessing

echo "Configuration:"
echo "  Case 2 (All Feedback): $CASE2_ALL_NAME (step $CASE2_ALL_GLOBAL_STEP)"
echo "  Case 2 (Single Feedback): $CASE2_SINGLE_NAME (step $CASE2_SINGLE_GLOBAL_STEP)"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Test size: $TEST_SIZE samples"
echo "  Seed: $SEED"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Merge Case 2 (All Feedback) checkpoint
echo "Step 1/7: Merging Case 2 (All Feedback) checkpoint..."
echo "====================================================="
bash helpsteer3/merge_checkpoint.sh "$CASE2_ALL_NAME" "$CASE2_ALL_GLOBAL_STEP"
echo ""

# Step 2: Merge Case 2 (Single Feedback) checkpoint
echo "Step 2/7: Merging Case 2 (Single Feedback) checkpoint..."
echo "========================================================"
bash helpsteer3/merge_checkpoint.sh "$CASE2_SINGLE_NAME" "$CASE2_SINGLE_GLOBAL_STEP"
echo ""

# Paths to merged models
CASE2_ALL_MODEL="checkpoints/helpsteer3_ppo_experiment/${CASE2_ALL_NAME}/global_step_${CASE2_ALL_GLOBAL_STEP}/actor/huggingface"
CASE2_SINGLE_MODEL="checkpoints/helpsteer3_ppo_experiment/${CASE2_SINGLE_NAME}/global_step_${CASE2_SINGLE_GLOBAL_STEP}/actor/huggingface"

# Step 3: Run inference on Case 2 (All Feedback)
echo "Step 3/7: Running inference for Case 2 (All Feedback)..."
echo "========================================================="
echo "Test size: $TEST_SIZE samples"
python helpsteer3/inference.py \
    --model_path "$CASE2_ALL_MODEL" \
    --output_file "$OUTPUT_DIR/case2_all_feedback_predictions.json" \
    --batch_size 8 \
    --max_new_tokens 768 \
    --test_size $TEST_SIZE \
    --seed $SEED
echo ""

# Step 4: Run inference on Case 2 (Single Feedback)
echo "Step 4/7: Running inference for Case 2 (Single Feedback)..."
echo "============================================================"
echo "Test size: $TEST_SIZE samples"
python helpsteer3/inference.py \
    --model_path "$CASE2_SINGLE_MODEL" \
    --output_file "$OUTPUT_DIR/case2_single_feedback_predictions.json" \
    --batch_size 8 \
    --max_new_tokens 768 \
    --test_size $TEST_SIZE \
    --seed $SEED
echo ""

# Download reward model if not already present
echo "Ensuring reward model is available..."
huggingface-cli download Jennny/llama3_help_rm
echo ""

# Step 5: Evaluate Case 2 (All Feedback)
echo "Step 5/7: Evaluating Case 2 (All Feedback)..."
echo "=============================================="
echo "Using reward model: Jennny/llama3_help_rm"
python helpsteer3/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case2_all_feedback_predictions.json" \
    --reward_model_path "Jennny/llama3_help_rm" \
    --output_file "$OUTPUT_DIR/case2_all_feedback_evaluation.json" \
    --batch_size 8
echo ""

# Step 6: Evaluate Case 2 (Single Feedback)
echo "Step 6/7: Evaluating Case 2 (Single Feedback)..."
echo "================================================="
echo "Using reward model: Jennny/llama3_help_rm"
python helpsteer3/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case2_single_feedback_predictions.json" \
    --reward_model_path "Jennny/llama3_help_rm" \
    --output_file "$OUTPUT_DIR/case2_single_feedback_evaluation.json" \
    --batch_size 8
echo ""

# Step 7: Compare results
echo "Step 7/7: Comparing results..."
echo "==============================="
python helpsteer3/compare_results.py \
    --case1_file "$OUTPUT_DIR/case2_all_feedback_evaluation.json" \
    --case2_file "$OUTPUT_DIR/case2_single_feedback_evaluation.json" \
    --output_file "$OUTPUT_DIR/case2_variants_comparison_report.json"
echo ""

echo "========================================="
echo "Case 2 Variants Evaluation Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  $OUTPUT_DIR/case2_all_feedback_predictions.json ($TEST_SIZE samples)"
echo "  $OUTPUT_DIR/case2_all_feedback_evaluation.json"
echo "  $OUTPUT_DIR/case2_single_feedback_predictions.json ($TEST_SIZE samples)"
echo "  $OUTPUT_DIR/case2_single_feedback_evaluation.json"
echo "  $OUTPUT_DIR/case2_variants_comparison_report.json"
echo ""
echo "Comparison:"
echo "  Case 2 (All Feedback):    Uses all feedback items from the list"
echo "  Case 2 (Single Feedback): Uses one randomly selected feedback item"
echo ""
echo "Test set:"
echo "  - $TEST_SIZE samples from HelpSteer3 validation split"
echo "  - Same samples used for both variants (seed=$SEED)"
echo ""
echo "Next steps:"
echo "  1. Review case2_variants_comparison_report.json for statistical results"
echo "  2. Check individual evaluation files for detailed scores"
echo "  3. Inspect predictions to understand model behavior differences"
echo "  4. Compare with W&B training metrics:"
echo "     - helpsteer3_case2_with_feedback_experimental (all feedback)"
echo "     - helpsteer3_case2_single_feedback_variant (single feedback)"
echo ""
echo "Research Question:"
echo "  Does providing all feedback items vs. one focused feedback"
echo "  make a difference in the quality of generated responses?"


