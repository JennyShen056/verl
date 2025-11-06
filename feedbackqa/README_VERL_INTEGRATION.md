# Using Trained Reward Model with verl PPO

This guide explains how to train a reward model from binary classification data and use it with verl's PPO training.

## Overview

The modified `rm_train.py` creates a **verl-compatible reward model** that:
- Uses `AutoModelForTokenClassification` (not SequenceClassification)
- Outputs a single scalar reward per token position
- Extracts the reward at the EOS token position
- Works seamlessly with verl's PPO training pipeline

## Key Differences from Standard Classification

| Aspect | Standard Classification | verl-Compatible RM |
|--------|------------------------|-------------------|
| Model Type | `AutoModelForSequenceClassification` | `AutoModelForTokenClassification` |
| Output | 2 logits (one per class) | 1 scalar per token |
| `num_labels` | 2 | 1 |
| Target Labels | Integer classes (0, 1) | Float rewards (0.0, 1.0) |
| Usage | Classification only | PPO reward signal |

## Step 1: Prepare Your Data

Your data files (`feedback_train_rm.json`, `feedback_valid_rm.json`, `feedback_test_rm.json`) should have this format:

```json
[
  {
    "question": "What is the capital of France?",
    "section_content": "The capital of France is Paris.",
    "label": 1
  },
  {
    "question": "What is 2+2?",
    "section_content": "The answer is 5.",
    "label": 0
  }
]
```

Where:
- `question`: The user's question
- `section_content`: The answer/response to evaluate
- `label`: 1 for good answer, 0 for bad answer

## Step 2: Train the Reward Model

```bash
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./feedback_qa_reward_model \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --max_length 1024
```

This will:
1. Load your binary classification data
2. Format using the model's chat template
3. Train a reward model compatible with verl
4. Save to `./feedback_qa_reward_model/final_model`

## Step 3: Verify the Trained Model

The trained model should have:
```python
# Load and verify
from transformers import AutoModelForTokenClassification, AutoConfig

config = AutoConfig.from_pretrained("./feedback_qa_reward_model/final_model")
print(config.num_labels)  # Should be 1
print(config.architectures)  # Should contain TokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    "./feedback_qa_reward_model/final_model"
)
# Model is ready for verl!
```

## Step 4: Prepare PPO Training Data

Your PPO training data should be in parquet format with at least:
- `prompts`: The questions/prompts
- `reward_model`: Dictionary with metadata (can include `ground_truth`, `data_source`, etc.)

Example:
```python
import pandas as pd

data = pd.DataFrame([
    {
        "prompts": "What is the capital of France?",
        "reward_model": {"ground_truth": "Paris", "data_source": "qa"}
    },
    # ... more examples
])

data.to_parquet("ppo_train.parquet")
```

## Step 5: Run PPO with Your Trained Reward Model

Use the example script:

```bash
bash feedbackqa/run_ppo_with_trained_rm.sh
```

Or customize your own:

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=$YOUR_PPO_TRAIN_DATA \
    data.val_files=$YOUR_PPO_TEST_DATA \
    actor_rollout_ref.model.path=$YOUR_POLICY_MODEL \
    reward_model.enable=True \
    reward_model.model.path=./feedback_qa_reward_model/final_model \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=16 \
    # ... other parameters
```

## How It Works in verl

When you enable the reward model in verl:

1. **Model Loading** (`verl/workers/fsdp_workers.py:1668`):
   ```python
   reward_module = AutoModelForTokenClassification.from_pretrained(
       pretrained_model_name_or_path="./feedback_qa_reward_model/final_model",
       config=model_config,  # num_labels=1
   )
   ```

2. **Inference** (`verl/workers/fsdp_workers.py:1776-1784`):
   ```python
   output = reward_module(input_ids, attention_mask, position_ids)
   rm_score = output.logits  # Shape: [batch_size, seq_len, 1]
   
   # Extract reward at EOS position
   eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)
   reward = rm_score[torch.arange(batch_size), eos_mask_idx]
   ```

3. **PPO Training** (`verl/trainer/ppo/ray_trainer.py:1154-1162`):
   ```python
   # Stage 1: Get neural network reward
   if self.use_rm:
       reward_tensor = self.rm_wg.compute_rm_score(batch)
       batch = batch.union(reward_tensor)  # Add rm_scores to batch
   
   # Stage 2: Optional function-based reward (can combine with rm_scores)
   reward_tensor, reward_extra_infos = compute_reward(batch, self.reward_fn)
   
   # Use rewards for advantage computation and policy update
   batch.batch["token_level_scores"] = reward_tensor
   ```

## Architecture Summary

```
Your Data (question + answer + label)
         ↓
    rm_train.py
         ↓
TokenClassification Model
  (num_labels=1, outputs scalar per token)
         ↓
Saved to ./feedback_qa_reward_model/final_model
         ↓
verl PPO Training
  - Loads model via fsdp_workers.py
  - Runs inference to get rm_scores
  - Uses scores for PPO updates
         ↓
Improved Policy Model
```

## Troubleshooting

### Issue: Model not loading in verl
**Solution**: Verify the model uses `AutoModelForTokenClassification` with `num_labels=1`

### Issue: Shape mismatch errors
**Solution**: Ensure your model outputs shape `[batch_size, seq_len, 1]`, not `[batch_size, 2]`

### Issue: Poor reward predictions
**Solutions**:
- Train for more epochs
- Use more training data
- Check data quality and label balance
- Tune learning rate and batch size

### Issue: OOM during PPO
**Solutions**:
- Reduce `reward_model.micro_batch_size_per_gpu`
- Enable `reward_model.model.fsdp_config.param_offload=True`
- Use gradient checkpointing
- Reduce sequence lengths

## Advanced: Combining with Function-Based Rewards

You can combine your trained RM with function-based rewards:

```python
# In your custom reward function
def custom_reward_fn(data_source, solution_str, ground_truth, extra_info):
    # Get the RM score if available
    rm_score = extra_info.get('rollout_reward_scores', {}).get('rm_scores', 0)
    
    # Add your custom logic
    if "correct_keyword" in solution_str:
        bonus = 0.5
    else:
        bonus = 0.0
    
    # Combine
    final_reward = rm_score + bonus
    return final_reward
```

## Files Created

- `rm_train.py`: Modified trainer (verl-compatible)
- `run_ppo_with_trained_rm.sh`: Example PPO script
- `README_VERL_INTEGRATION.md`: This guide

## References

- verl documentation: See `examples/ppo_trainer/run_qwen2-7b_rm.sh`
- Worker implementation: `verl/workers/fsdp_workers.py`
- PPO trainer: `verl/trainer/ppo/ray_trainer.py`

