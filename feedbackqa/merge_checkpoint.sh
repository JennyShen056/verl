#!/bin/bash
# Merge PPO Checkpoint to HuggingFace Format
# ==========================================
# Converts FSDP checkpoint to standard HuggingFace format for inference

set -e

if [ "$#" -lt 2 ]; then
    echo "Usage: bash merge_checkpoint.sh <case_name> <global_step>"
    echo ""
    echo "Examples:"
    echo "  bash merge_checkpoint.sh case1_question_only_baseline 1"
    echo "  bash merge_checkpoint.sh case2_with_feedback_experimental 5"
    echo ""
    echo "Available checkpoints:"
    ls -d checkpoints/feedback_qa_experiment/*/global_step_* 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

CASE_NAME=$1
GLOBAL_STEP=$2

PROJECT_NAME="feedback_qa_experiment"
CHECKPOINT_DIR="checkpoints/${PROJECT_NAME}/${CASE_NAME}/global_step_${GLOBAL_STEP}"
ACTOR_DIR="${CHECKPOINT_DIR}/actor"
TARGET_DIR="${ACTOR_DIR}/huggingface"

echo "========================================="
echo "Merging Checkpoint to HuggingFace Format"
echo "========================================="
echo "Case: ${CASE_NAME}"
echo "Global Step: ${GLOBAL_STEP}"
echo "Source: ${ACTOR_DIR}"
echo "Target: ${TARGET_DIR}"
echo "========================================="

# Check if checkpoint exists
if [ ! -d "$ACTOR_DIR" ]; then
    echo "ERROR: Checkpoint not found at $ACTOR_DIR"
    echo ""
    echo "Available checkpoints:"
    ls -d checkpoints/feedback_qa_experiment/*/global_step_* 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Merge checkpoint
echo "Merging checkpoint..."
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"

echo ""
echo "========================================="
echo "Merge Complete!"
echo "========================================="
echo "HuggingFace model saved to: $TARGET_DIR"
echo ""
echo "You can now use this for inference:"
echo "  python feedbackqa/inference.py --model_path $TARGET_DIR"

