# Reward Model Format: num_labels=1 vs num_labels=2

## Summary of Changes

Your `rm_train.py` has been updated to produce **`num_labels=1`** format (scalar regression) instead of **`num_labels=2`** format (binary classification). This makes it compatible with verl's PPO training.

## Why the Difference?

### Your Original Code (num_labels=2)
- **Approach**: Binary classification
- **Output**: 2 logits `[logit_class_0, logit_class_1]`
- **Shape**: `[2, 4096]` - two classification heads
- **Loss**: Cross-entropy loss
- **Use case**: Traditional text classification tasks
- **Problem**: verl expects a single scalar reward score, not class logits

### RLHFlow Code (num_labels=1)
- **Approach**: Scalar regression (Bradley-Terry preference learning)
- **Output**: 1 scalar reward score per input
- **Shape**: `[1, 4096]` - single regression head
- **Loss**: `-log(sigmoid(reward_chosen - reward_rejected))`
- **Use case**: Preference learning with paired data (chosen vs rejected)
- **Why it works**: Outputs a continuous reward score that verl can use directly

### Your Updated Code (num_labels=1)
- **Approach**: Scalar regression (adapted for binary labels)
- **Output**: 1 scalar reward score per input
- **Shape**: `[1, 4096]` - single regression head
- **Loss**: MSE loss (automatic with `problem_type="regression"`)
- **Use case**: Binary relevance labels without preference pairs
- **Why it works**: Outputs a continuous reward score like RLHFlow, but trained on binary labels

## Key Changes Made

### 1. Model Configuration
```python
# Before (num_labels=2)
config.num_labels = 2
config.problem_type = "single_label_classification"

# After (num_labels=1)
config.num_labels = 1
config.problem_type = "regression"
```

### 2. Label Processing
```python
# Before: Integer labels for classification
tokenized["labels"] = [int(label) for label in examples["label"]]
# Labels: [0, 1, 0, 1, ...]

# After: Float labels for regression
tokenized["labels"] = [float(label) * 2.0 - 1.0 for label in examples["label"]]
# Labels: [-1.0, 1.0, -1.0, 1.0, ...]
# Label 0 (not relevant) → -1.0
# Label 1 (relevant) → 1.0
```

### 3. Predictions
```python
# Before: Softmax over 2 logits
probs = F.softmax(predictions, dim=-1)[:, 1]
predicted_classes = (probs > 0.5).astype(int)

# After: Threshold scalar predictions
predictions = predictions.squeeze(-1)  # Shape: [batch_size]
predicted_classes = (predictions > 0.0).astype(int)
```

### 4. Metrics
The metrics computation now:
1. Takes scalar predictions (not 2 logits)
2. Converts predictions to binary classes using threshold 0.0
3. Converts labels back to {0, 1} for evaluation metrics
4. Uses continuous predictions for AUC score

## How It Works with verl

When verl loads your reward model:

```python
# verl calls the model like this:
reward_scores = model(input_ids, attention_mask)
# With num_labels=1: reward_scores.shape = [batch_size, 1]
# With num_labels=2: reward_scores.shape = [batch_size, 2] ← WRONG!
```

verl expects a **single scalar reward** for each input to use in PPO's advantage calculation:

```python
advantages = rewards - baseline_values
```

With `num_labels=2`, verl would get 2 values per input, causing shape mismatches.

## Training Differences

| Aspect | Binary Classification (num_labels=2) | Scalar Regression (num_labels=1) |
|--------|--------------------------------------|----------------------------------|
| **Output** | 2 logits | 1 scalar |
| **Loss** | Cross-entropy | MSE (or custom) |
| **Activation** | Softmax | None (raw score) |
| **Range** | [0, 1] (probabilities) | Unbounded (-∞, +∞) |
| **Interpretation** | Class probabilities | Reward score |
| **verl Compatible** | ❌ No | ✅ Yes |

## Why RLHFlow Uses Bradley-Terry

The RLHFlow code trains on **preference pairs** (chosen vs rejected responses):

```python
# Data format:
{
  "chosen": "Good response",
  "rejected": "Bad response"
}

# Loss:
loss = -log(sigmoid(reward_chosen - reward_rejected))
```

This directly learns to predict higher rewards for preferred responses.

## Why Your Approach Works Too

You don't have preference pairs, just binary labels:

```python
# Your data format:
{
  "section_content": "Some text",
  "label": 1  # relevant (1) or not (0)
}
```

By using regression with `num_labels=1`:
- Label 0 → target reward = -1.0 (low reward for not relevant)
- Label 1 → target reward = +1.0 (high reward for relevant)
- Model learns to output higher scores for relevant content
- verl can use these scores directly in PPO

## Verification

After retraining with the updated code, verify the model shape:

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "./feedback_qa_reward_model/final_model"
)

print(f"Config num_labels: {model.config.num_labels}")
# Should output: Config num_labels: 1

print(f"Classifier output features: {model.classifier.out_features}")  
# Should output: Classifier output features: 1
```

## References

- RLHFlow Bradley-Terry RM: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/bradley-terry-rm/llama3_8B_rm.py
- Working example: `sfairXC/FsfairX-LLaMA3-RM-v0.1` (uses num_labels=1)

## Next Steps

1. **Retrain** your reward model with the updated `rm_train.py`
2. **Verify** the output shape is `[1, 4096]` using the verification code above
3. **Test** with verl PPO training to ensure compatibility
4. **Compare** performance with your previous num_labels=2 model

The updated code maintains all your evaluation metrics (accuracy, precision, recall, F1, AUC) while producing the correct output format for verl.

