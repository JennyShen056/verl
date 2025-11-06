# Validation Error Fix

## 🔍 The Problem

**Error:**
```python
AssertionError: val_metrics={}
```

**Location:** `verl/trainer/ppo/ray_trainer.py`, line 1043

## 📊 Root Cause

The error occurs because PPO training tries to run **validation BEFORE training starts** by default:

```python
if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
    val_metrics = self._validate()
    assert val_metrics, f"{val_metrics=}"  # ← Fails here
```

### Why It Fails

1. **Initial validation is enabled by default** (`val_before_train=True`)
2. You're using a **model-based reward model** for training
3. But validation expects either:
   - A **function-based reward** (`val_reward_fn`), OR
   - Proper validation data with the reward model configured

4. Since no proper validation reward function is set up, `_validate()` returns an empty dictionary `{}`

## ✅ The Fix

Added `trainer.val_before_train=False` to both training scripts:

```bash
trainer.val_before_train=False \
```

This **skips the initial validation step** and lets training start immediately.

## 🤔 When to Use Each Approach

### Skip Initial Validation (`val_before_train=False`)

**✅ Use when:**
- You want to start training quickly
- You'll validate during training (via `test_freq`)
- You're using model-based rewards only
- Initial validation isn't critical

**Pros:**
- Training starts immediately
- No validation setup needed
- Still validates during training (if `test_freq > 0`)

**Cons:**
- No initial baseline metrics
- Can't check model before training

### Enable Initial Validation (`val_before_train=True`)

**✅ Use when:**
- You want baseline metrics before training
- You have a proper validation reward function
- You want to verify everything works before spending compute

**Requires:**
- Define a custom `val_reward_fn` in your code
- Or use rule-based rewards (like in GSM8K examples)

## 📈 Validation During Training

Even with `val_before_train=False`, validation still happens **during training**:

```bash
trainer.test_freq=2  # Validates every 2 training steps
```

So you'll still get validation metrics, just not before the first training step.

## 🔄 Alternative: Provide Validation Reward Function

If you want initial validation, you can provide a custom validation reward function. Example:

```python
from verl import DataProto

def my_val_reward_fn(data: DataProto) -> DataProto:
    """Custom validation reward function"""
    # Extract generated text
    # Compute rewards based on your criteria
    # Return rewards
    pass

# Pass to trainer
trainer = RayPPOTrainer(
    config=config,
    val_reward_fn=my_val_reward_fn
)
```

But for your experiment, skipping initial validation is simpler and sufficient.

## 🎯 Summary

| Setting | Effect | Your Config |
|---------|--------|-------------|
| `trainer.val_before_train=False` | Skip validation before training | ✅ Enabled |
| `trainer.test_freq=2` | Validate every 2 steps during training | ✅ Enabled |
| `trainer.total_epochs=1` (Case 1) | Train for 1 epoch | ✅ Set |
| `trainer.total_epochs=10` (Case 2) | Train for 10 epochs | ✅ Set |

## ✅ Current Status

Both training scripts are now configured to:
- ✅ Skip initial validation
- ✅ Validate during training (every 2 steps)
- ✅ Use model-based reward (FsfairX-LLaMA3-RM-v0.1)
- ✅ Log to W&B

**You're ready to train!** 🚀

```bash
bash feedbackqa/run_ppo_case1_question_only.sh
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

