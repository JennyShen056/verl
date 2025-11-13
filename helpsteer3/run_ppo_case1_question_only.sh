#!/bin/bash
# PPO Training Script for HelpSteer3 Case 1: Question Only (Baseline)
# ====================================================================
# This is the baseline experiment where the model receives ONLY the conversation context
# and must learn to generate good responses from scratch using RM feedback.

set -x

# Paths
DATA_DIR="$HOME/data/helpsteer3_ppo/case1_question_only"
POLICY_MODEL="meta-llama/Llama-3.1-8B-Instruct"

# Training data (preprocessed with case1 script)
TRAIN_DATA="$DATA_DIR/train.parquet"
VALID_DATA="$DATA_DIR/validation.parquet"

# Download reward model in background
huggingface-cli download Jennny/llama3_help_rm --local-dir $HOME/models/llama3_help_rm &

echo "========================================="
echo "PPO Training - HelpSteer3 Case 1: Question Only"
echo "========================================="
echo "Experiment: Baseline - no feedback examples"
echo "Input format: Conversation context only"
echo "RM scores: Conversation + Generated Response"
echo "Dataset: HelpSteer3 feedback subset"
echo "========================================="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VALID_DATA" \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=768 \
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
    reward_model.model.path="$HOME/models/llama3_help_rm" \
    reward_model.model.input_tokenizer="meta-llama/Llama-3.1-8B-Instruct" \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=16 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='helpsteer3_ppo_experiment' \
    trainer.experiment_name='helpsteer3_case1_question_only_baseline' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=5 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 $@

echo ""
echo "========================================="
echo "HelpSteer3 Case 1 Training Complete!"
echo "========================================="
echo "Model saved to default checkpoint directory"
echo "Next: Train Case 2 and compare results"

