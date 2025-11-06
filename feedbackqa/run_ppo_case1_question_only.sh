#!/bin/bash
# PPO Training Script for Case 1: Question Only (Baseline)
# ==========================================================
# This is the baseline experiment where the model receives ONLY the question
# and must learn to generate good answers from scratch using RM feedback.

set -x

# Paths
TRAINED_RM_PATH="./feedback_qa_reward_model/final_model"
DATA_DIR="$HOME/data/feedback_qa_ppo/case1_question_only"
POLICY_MODEL=meta-llama/Llama-3.1-8B-Instruct"

# Training data (preprocessed with case1 script)
TRAIN_DATA="$DATA_DIR/train.parquet"
VALID_DATA="$DATA_DIR/valid.parquet"

# Check if data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA"
    echo "Please run: python feedbackqa/preprocess_ppo_case1_question_only.py"
    exit 1
fi

if [ ! -d "$TRAINED_RM_PATH" ]; then
    echo "ERROR: Trained reward model not found at $TRAINED_RM_PATH"
    echo "Please run: python feedbackqa/rm_train.py"
    exit 1
fi

echo "========================================="
echo "PPO Training - Case 1: Question Only"
echo "========================================="
echo "Experiment: Baseline (no feedback examples)"
echo "Input format: Question only"
echo "RM scores: Question + Generated Answer"
echo "========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VALID_DATA" \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=565 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$POLICY_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="$POLICY_MODEL" \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=16 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    reward_model.enable=True \
    reward_model.model.path="$TRAINED_RM_PATH" \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='feedback_qa_experiment' \
    trainer.experiment_name='case1_question_only_baseline' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 $@

echo ""
echo "========================================="
echo "Case 1 Training Complete!"
echo "========================================="
echo "Model saved to default checkpoint directory"
echo "Next: Train Case 2 and compare results"

