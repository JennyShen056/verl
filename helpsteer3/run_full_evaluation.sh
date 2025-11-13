#!/bin/bash
# Complete Evaluation Pipeline for HelpSteer3
# ============================================
# Merges checkpoints, runs inference, evaluates, and compares results
#
# Usage:
#   bash helpsteer3/run_full_evaluation.sh
#
# Note: Edit CASE1_GLOBAL_STEP and CASE2_GLOBAL_STEP below to evaluate different checkpoints

set -e

echo "========================================="
echo "HelpSteer3 Complete Evaluation Pipeline"
echo "========================================="
echo "This script will:"
echo "  1. Merge Case 1 checkpoint to HuggingFace format"
echo "  2. Merge Case 2 checkpoint to HuggingFace format"
echo "  3. Run inference on test set for both cases"
echo "  4. Evaluate using reward model (Jennny/llama3_help_rm)"
echo "  5. Compare results statistically"
echo "========================================="
echo ""

# Configuration
CASE1_NAME="helpsteer3_case1_question_only_baseline"
CASE2_NAME="helpsteer3_case2_with_feedback_experimental"
CASE1_GLOBAL_STEP=115  # Global step for Case 1
CASE2_GLOBAL_STEP=195  # Global step for Case 2
OUTPUT_DIR="helpsteer3/outputs"
TEST_SIZE=500  # Number of test samples (same as preprocessing)
SEED=42  # Same seed as preprocessing

echo "Configuration:"
echo "  Case 1: $CASE1_NAME (step $CASE1_GLOBAL_STEP)"
echo "  Case 2: $CASE2_NAME (step $CASE2_GLOBAL_STEP)"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Test size: $TEST_SIZE samples"
echo "  Seed: $SEED"
echo ""

mkdir -p "$OUTPUT_DIR"

# Step 1: Merge Case 1 checkpoint
echo "Step 1/5: Merging Case 1 checkpoint..."
echo "======================================="
python3 -m verl.utils.fs copy \
    "checkpoints/helpsteer3_ppo_experiment/${CASE1_NAME}/global_step_${CASE1_GLOBAL_STEP}/actor/default" \
    "checkpoints/helpsteer3_ppo_experiment/${CASE1_NAME}/global_step_${CASE1_GLOBAL_STEP}/actor/huggingface"

python3 -m verl.utils.model_merge \
    --source "checkpoints/helpsteer3_ppo_experiment/${CASE1_NAME}/global_step_${CASE1_GLOBAL_STEP}/actor/default" \
    --target "checkpoints/helpsteer3_ppo_experiment/${CASE1_NAME}/global_step_${CASE1_GLOBAL_STEP}/actor/huggingface"
echo ""

# Step 2: Merge Case 2 checkpoint
echo "Step 2/5: Merging Case 2 checkpoint..."
echo "======================================="
python3 -m verl.utils.fs copy \
    "checkpoints/helpsteer3_ppo_experiment/${CASE2_NAME}/global_step_${CASE2_GLOBAL_STEP}/actor/default" \
    "checkpoints/helpsteer3_ppo_experiment/${CASE2_NAME}/global_step_${CASE2_GLOBAL_STEP}/actor/huggingface"

python3 -m verl.utils.model_merge \
    --source "checkpoints/helpsteer3_ppo_experiment/${CASE2_NAME}/global_step_${CASE2_GLOBAL_STEP}/actor/default" \
    --target "checkpoints/helpsteer3_ppo_experiment/${CASE2_NAME}/global_step_${CASE2_GLOBAL_STEP}/actor/huggingface"
echo ""

# Paths to merged models
CASE1_MODEL="checkpoints/helpsteer3_ppo_experiment/${CASE1_NAME}/global_step_${CASE1_GLOBAL_STEP}/actor/huggingface"
CASE2_MODEL="checkpoints/helpsteer3_ppo_experiment/${CASE2_NAME}/global_step_${CASE2_GLOBAL_STEP}/actor/huggingface"

# Step 3: Run inference on Case 1
echo "Step 3/5: Running inference for Case 1..."
echo "=========================================="
echo "Test size: $TEST_SIZE samples"
python helpsteer3/inference.py \
    --model_path "$CASE1_MODEL" \
    --output_file "$OUTPUT_DIR/case1_predictions.json" \
    --batch_size 8 \
    --max_new_tokens 768 \
    --test_size $TEST_SIZE \
    --seed $SEED
echo ""

# Step 4: Run inference on Case 2
echo "Step 4/5: Running inference for Case 2..."
echo "=========================================="
echo "Test size: $TEST_SIZE samples"
python helpsteer3/inference.py \
    --model_path "$CASE2_MODEL" \
    --output_file "$OUTPUT_DIR/case2_predictions.json" \
    --batch_size 8 \
    --max_new_tokens 768 \
    --test_size $TEST_SIZE \
    --seed $SEED
echo ""

# Download reward model if not already present
echo "Ensuring reward model is available..."
huggingface-cli download Jennny/llama3_help_rm

# Step 5: Evaluate Case 1
echo "Step 5/7: Evaluating Case 1..."
echo "==============================="
echo "Using reward model: Jennny/llama3_help_rm"
python helpsteer3/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case1_predictions.json" \
    --reward_model_path "Jennny/llama3_help_rm" \
    --output_file "$OUTPUT_DIR/case1_evaluation.json" \
    --batch_size 8
echo ""

# Step 6: Evaluate Case 2
echo "Step 6/7: Evaluating Case 2..."
echo "==============================="
echo "Using reward model: Jennny/llama3_help_rm"
python helpsteer3/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case2_predictions.json" \
    --reward_model_path "Jennny/llama3_help_rm" \
    --output_file "$OUTPUT_DIR/case2_evaluation.json" \
    --batch_size 8
echo ""

# Step 7: Compare results
echo "Step 7/7: Comparing results..."
echo "==============================="
python helpsteer3/compare_results.py \
    --case1_file "$OUTPUT_DIR/case1_evaluation.json" \
    --case2_file "$OUTPUT_DIR/case2_evaluation.json" \
    --output_file "$OUTPUT_DIR/comparison_report.json"
echo ""

echo "========================================="
echo "Evaluation Pipeline Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  $OUTPUT_DIR/case1_predictions.json ($TEST_SIZE samples)"
echo "  $OUTPUT_DIR/case1_evaluation.json"
echo "  $OUTPUT_DIR/case2_predictions.json ($TEST_SIZE samples)"
echo "  $OUTPUT_DIR/case2_evaluation.json"
echo "  $OUTPUT_DIR/comparison_report.json"
echo ""
echo "Test set:"
echo "  - $TEST_SIZE samples from HelpSteer3 validation split"
echo "  - Same samples used for both cases (seed=$SEED)"
echo ""
echo "Next steps:"
echo "  1. Review comparison_report.json for statistical results"
echo "  2. Check individual evaluation files for detailed scores"
echo "  3. Inspect predictions to understand model behavior"
echo "  4. Compare with W&B training metrics"

