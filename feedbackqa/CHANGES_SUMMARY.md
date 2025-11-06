# Summary of Changes to rm_train.py for verl Compatibility

## Problem
The original script trained a binary classification model using `AutoModelForSequenceClassification`, which outputs 2 logits (one per class). However, verl's PPO training expects a reward model that:
1. Uses `AutoModelForTokenClassification`
2. Outputs a **single scalar score** per token position
3. Has `num_labels=1` in the config

## Changes Made

### 1. Import Statement (Line 13-21)
**Before:**
```python
from transformers import (
    AutoModelForSequenceClassification,
    ...
)
```

**After:**
```python
from transformers import (
    AutoModelForTokenClassification,  # Changed for verl compatibility
    ...
)
```

**Why:** verl expects `AutoModelForTokenClassification` which outputs per-token scores.

### 2. Model Configuration (Line 298-311)
**Before:**
```python
config.num_labels = 2  # Binary classification: 2 classes (0, 1)
config.problem_type = "single_label_classification"
config.classifier_dropout = 0.1

self.model = AutoModelForSequenceClassification.from_pretrained(...)
```

**After:**
```python
config.num_labels = 1  # Single scalar output for reward score
config.classifier_dropout = 0.0  # verl uses 0.0 dropout

self.model = AutoModelForTokenClassification.from_pretrained(...)
```

**Why:** 
- `num_labels=1` → single reward score instead of 2 class probabilities
- `classifier_dropout=0.0` → matches verl's FSDP worker configuration
- Changed model class to match verl's expectations

### 3. Label Format (Line 341-343)
**Before:**
```python
# Ensure labels are integers (0 or 1) for classification
tokenized["labels"] = [int(label) for label in examples["label"]]
```

**After:**
```python
# Convert binary labels (0, 1) to float reward scores
# verl will use the score at EOS position as the reward
tokenized["labels"] = [float(label) for label in examples["label"]]
```

**Why:** Reward models output continuous scores, not discrete classes.

### 4. Metrics Computation (Line 346-408)
**Before:**
```python
# For binary classification, predictions are logits of shape [batch_size, 2]
# We take the softmax and use the probability of class 1
probs = F.softmax(torch.from_numpy(predictions), dim=-1)[:, 1].numpy()
predicted_classes = (probs > 0.5).astype(int)
```

**After:**
```python
# For reward model with num_labels=1, predictions are of shape [batch_size, seq_len, 1]
# Extract the score at the EOS position (last valid token)
if len(predictions.shape) == 3:
    scores = predictions[:, -1, 0]  # Take last position score
elif len(predictions.shape) == 2:
    scores = predictions[:, -1]
else:
    scores = predictions.squeeze()

# Apply sigmoid to map to [0, 1] probability range
probs = torch.sigmoid(torch.from_numpy(scores)).numpy()
predicted_classes = (probs > 0.5).astype(int)
```

**Why:** TokenClassification outputs per-token scores, need to extract the relevant one.

### 5. Training Summary (Line 513-535)
**Before:**
```python
"classification_type": "binary",
"model_config": {
    "num_labels": 2,
    "problem_type": "single_label_classification",
    ...
}
```

**After:**
```python
"model_type": "token_classification_reward_model",
"verl_compatible": True,
"model_config": {
    "num_labels": 1,
    "architecture": "AutoModelForTokenClassification",
    "output_type": "single_scalar_reward_per_token",
    ...
}
```

**Why:** Document that the model is verl-compatible.

### 6. Docstrings and Comments
Updated all docstrings to reflect the new purpose:
- Class docstring mentions verl compatibility
- Method docstrings explain reward model behavior
- Comments clarify that EOS token position is used for rewards

## How It Works with verl

```
Training (rm_train.py):
┌─────────────────────────────────────────┐
│ Input: question + answer + label (0/1)  │
│ Model: TokenClassification (num_labels=1)│
│ Output: Scalar score per token position │
│ Loss: MSE/BCE at EOS position           │
└─────────────────────────────────────────┘
                    ↓
            Saved Model
                    ↓
Inference (verl/workers/fsdp_workers.py):
┌─────────────────────────────────────────┐
│ 1. Load via AutoModelForTokenClassification│
│ 2. Forward: get logits [batch, seq, 1] │
│ 3. Extract: score at EOS position      │
│ 4. Return: single scalar reward per seq│
└─────────────────────────────────────────┘
                    ↓
PPO Training (verl/trainer/ppo/ray_trainer.py):
┌─────────────────────────────────────────┐
│ 1. Call rm_wg.compute_rm_score(batch)  │
│ 2. Get rm_scores in batch              │
│ 3. Use for advantage computation       │
│ 4. Update policy via PPO               │
└─────────────────────────────────────────┘
```

## Verification

After training, verify your model:
```bash
python feedbackqa/verify_trained_model.py --model_path ./feedback_qa_reward_model/final_model
```

This will check:
- ✓ `num_labels=1`
- ✓ Uses `AutoModelForTokenClassification`
- ✓ Output shape is `[batch_size, seq_len, 1]`
- ✓ Can extract reward at EOS position

## Usage

1. **Train the reward model:**
```bash
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./feedback_qa_reward_model
```

2. **Verify it works:**
```bash
python feedbackqa/verify_trained_model.py
```

3. **Use in PPO:**
```bash
bash feedbackqa/run_ppo_with_trained_rm.sh
```

Or in your config:
```bash
reward_model.enable=True \
reward_model.model.path=./feedback_qa_reward_model/final_model
```

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `rm_train.py` | **Modified** | Main training script (verl-compatible) |
| `run_ppo_with_trained_rm.sh` | **New** | Example PPO script |
| `verify_trained_model.py` | **New** | Verification tool |
| `README_VERL_INTEGRATION.md` | **New** | Complete guide |
| `CHANGES_SUMMARY.md` | **New** | This file |

## Key Takeaway

**The model now outputs a single scalar reward per token position (compatible with verl) instead of 2 class probabilities (standard classification).**

This makes it directly usable as a reward signal in verl's PPO training pipeline without any additional conversion or wrapping!

