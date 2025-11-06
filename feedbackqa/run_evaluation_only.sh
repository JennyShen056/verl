#!/bin/bash
# Evaluation Only - Run Steps 5-7
# ================================
# This script assumes you've already:
#   - Merged checkpoints (steps 1-2)
#   - Run inference (steps 3-4)
#   - Have prediction files in outputs/ directory
#
# Usage:
#   bash feedbackqa/run_evaluation_only.sh

set -e

echo "========================================="
echo "Evaluation Pipeline (Steps 5-7 Only)"
echo "========================================="
echo ""

# Configuration
OUTPUT_DIR="outputs"
REWARD_MODEL="Jennny/qa_rm"  # Change this to your reward model path

echo "Configuration:"
echo "  Reward Model: $REWARD_MODEL"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Check if prediction files exist
if [ ! -f "$OUTPUT_DIR/case1_predictions.json" ]; then
    echo "❌ Error: $OUTPUT_DIR/case1_predictions.json not found!"
    echo "   Please run inference (steps 3-4) first."
    exit 1
fi

if [ ! -f "$OUTPUT_DIR/case2_predictions.json" ]; then
    echo "❌ Error: $OUTPUT_DIR/case2_predictions.json not found!"
    echo "   Please run inference (steps 3-4) first."
    exit 1
fi

echo "✓ Found prediction files"
echo ""

# Step 5: Evaluate Case 1
echo "Step 5/7: Evaluating Case 1..."
echo "==============================="
python feedbackqa/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case1_predictions.json" \
    --reward_model_path "$REWARD_MODEL" \
    --output_file "$OUTPUT_DIR/case1_evaluation.json" \
    --batch_size 8
echo ""

# Step 6: Evaluate Case 2
echo "Step 6/7: Evaluating Case 2..."
echo "==============================="
python feedbackqa/evaluate.py \
    --predictions_file "$OUTPUT_DIR/case2_predictions.json" \
    --reward_model_path "$REWARD_MODEL" \
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
echo "Evaluation Complete!"
echo "========================================="
echo ""
echo "Results saved to:"
echo "  $OUTPUT_DIR/case1_evaluation.json"
echo "  $OUTPUT_DIR/case2_evaluation.json"
echo "  $OUTPUT_DIR/comparison_report.json"
echo ""
echo "To view the comparison:"
echo "  cat $OUTPUT_DIR/comparison_report.json | jq"

