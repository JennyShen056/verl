#!/bin/bash
# Monitor validation metrics during training
# Usage: bash helpsteer3/monitor_validation.sh <checkpoint_dir> <validation_data> <wandb_run_id>

set -e

CHECKPOINT_DIR=$1
VALIDATION_DATA=$2
WANDB_RUN_ID=$3

if [ -z "$CHECKPOINT_DIR" ] || [ -z "$VALIDATION_DATA" ]; then
    echo "Usage: bash helpsteer3/monitor_validation.sh <checkpoint_dir> <validation_data> [wandb_run_id]"
    echo ""
    echo "Example:"
    echo "  bash helpsteer3/monitor_validation.sh \\"
    echo "    checkpoints/helpsteer3_ppo_experiment/helpsteer3_case1_question_only_baseline \\"
    echo "    ~/data/helpsteer3_ppo/case1_question_only/validation.parquet \\"
    echo "    abc123xyz"
    exit 1
fi

echo "========================================="
echo "Validation Monitoring"
echo "========================================="
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Validation data: $VALIDATION_DATA"
echo "Wandb run ID: ${WANDB_RUN_ID:-None (will create new run)}"
echo "========================================="

# Find all checkpoints
CHECKPOINTS=$(find "$CHECKPOINT_DIR" -type d -name "global_step_*" | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo "No checkpoints found in $CHECKPOINT_DIR"
    exit 1
fi

echo ""
echo "Found checkpoints:"
echo "$CHECKPOINTS"
echo ""

# Process each checkpoint
for CHECKPOINT_PATH in $CHECKPOINTS; do
    # Extract step number
    STEP=$(basename "$CHECKPOINT_PATH" | sed 's/global_step_//')
    
    echo "========================================="
    echo "Processing checkpoint at step $STEP"
    echo "========================================="
    
    # Check if huggingface model exists
    HF_MODEL="$CHECKPOINT_PATH/actor/huggingface"
    if [ ! -d "$HF_MODEL" ]; then
        echo "⚠️  HuggingFace model not found at $HF_MODEL"
        echo "   Merging checkpoint..."
        
        # Merge checkpoint to HF format
        python3 -m verl.utils.fs copy \
            "$CHECKPOINT_PATH/actor/default" \
            "$HF_MODEL"
        
        python3 -m verl.utils.model_merge \
            --source "$CHECKPOINT_PATH/actor/default" \
            --target "$HF_MODEL"
        
        echo "✓ Checkpoint merged"
    fi
    
    # Generate responses and log to wandb
    if [ -n "$WANDB_RUN_ID" ]; then
        # Resume existing run
        python helpsteer3/log_responses_to_wandb.py \
            --model_path "$HF_MODEL" \
            --data_file "$VALIDATION_DATA" \
            --wandb_run_id "$WANDB_RUN_ID" \
            --step "$STEP" \
            --num_samples 20 \
            --split validation
    else
        # Create new run per checkpoint
        python helpsteer3/log_responses_to_wandb.py \
            --model_path "$HF_MODEL" \
            --data_file "$VALIDATION_DATA" \
            --wandb_name "$(basename $CHECKPOINT_DIR)_validation" \
            --step "$STEP" \
            --num_samples 20 \
            --split validation
    fi
    
    echo ""
done

echo "========================================="
echo "Validation monitoring complete!"
echo "========================================="

