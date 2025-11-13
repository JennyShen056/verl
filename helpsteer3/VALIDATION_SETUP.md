# Validation Metrics with Reward Model - Setup Guide

## What Was Added

I've enabled validation metrics during PPO training even when using reward models. Here's what was implemented:

### 1. Custom Validation Reward Function (`validation_reward_fn.py`)

A custom reward function that:
- ✅ Loads and caches the reward model (only loads once)
- ✅ Takes validation batches and computes reward scores
- ✅ Returns results in the format VERL expects
- ✅ Provides additional validation metrics (mean, std, max, min)

### 2. Updated Training Scripts

Both `run_ppo_case1_question_only.sh` and `run_ppo_case2_with_feedback.sh` now include:

```bash
# NEW: Custom reward function for validation
custom_reward_function.path="helpsteer3/validation_reward_fn.py"
custom_reward_function.name="compute_score"

# NEW: Return raw chat for proper prompt handling
data.return_raw_chat=True

# UPDATED: Increased response length (was 768, now 1024)
data.max_response_length=1024
```

## How It Works

### Training Flow

1. **Training**: Uses the fast reward model worker (FSDP-based)
   - Scores training batches efficiently
   - Updates policy and critic

2. **Validation** (every `test_freq` steps): Uses the custom reward function
   - Loads reward model once (cached)
   - Scores validation batches
   - Logs metrics to W&B

### Dual Reward System

```
Training:
  reward_model.enable=True → Uses FSDP reward model worker (fast, distributed)
  
Validation:
  custom_reward_function → Uses standalone reward model (simple, CPU/GPU)
```

This gives you the best of both worlds:
- Fast training with optimized reward model
- Easy validation with straightforward reward scoring

## Validation Metrics You'll See

### In W&B Dashboard

After enabling validation, you'll see these new metrics every `test_freq` steps:

**Main Metrics:**
- `test/reward/mean` - Average validation reward
- `test/reward/std` - Standard deviation of validation rewards
- `test/reward/max` - Maximum validation reward
- `test/reward/min` - Minimum validation reward

**By Data Source** (if applicable):
- `test/helpsteer3_feedback/reward/mean`
- `test/helpsteer3_feedback/reward/max`
- etc.

**Sample Outputs:**
- Random validation samples and their scores
- Logged to console and W&B

### Example Console Output

```
========================================
Running validation at step 10...
========================================
Loading validation reward model from Jennny/llama3_help_rm...
Validation reward model loaded on cuda:0

Processing validation batch 1/5...
Processing validation batch 2/5...
...

Validation Results:
  Mean reward: 2.45
  Std dev: 1.23
  Max reward: 5.67
  Min reward: -0.34
  
Sample outputs:
  [1] Input: User: How do I... 
      Output: To accomplish that, you should...
      Score: 3.21
  [2] Input: User: What is...
      Output: It is a concept that...
      Score: 2.87
```

## Key Changes Summary

| Setting | Old Value | New Value | Reason |
|---------|-----------|-----------|--------|
| `max_response_length` | 768 | 1024 | 99.6% were hitting limit |
| `custom_reward_function.path` | ❌ None | ✅ `validation_reward_fn.py` | Enable validation |
| `data.return_raw_chat` | ❌ False | ✅ True | Proper prompt handling |

## Usage

### Running Training with Validation

```bash
# Case 1: Question Only
bash helpsteer3/run_ppo_case1_question_only.sh

# Case 2: With Feedback
bash helpsteer3/run_ppo_case2_with_feedback.sh
```

### Monitoring in W&B

1. Go to your W&B project: `helpsteer3_ppo_experiment`
2. Look for metrics under `test/` prefix:
   - `test/reward/mean` - Main validation metric
   - `test/reward/*` - Other validation statistics

3. Compare training vs validation:
   - `critic/rewards/mean` (training)
   - `test/reward/mean` (validation)

### Adjusting Validation Frequency

```bash
# Validate more often (every step)
trainer.test_freq=1

# Validate less often (every 10 steps)
trainer.test_freq=10

# Default (every 2 steps)
trainer.test_freq=2
```

## Performance Considerations

### Memory Usage

The validation reward model:
- ✅ Loads once and is cached
- ✅ Uses bfloat16 precision
- ✅ Runs on a single GPU
- ⚠️ Adds ~8-16GB GPU memory overhead

If you run out of memory, you can:
1. Reduce validation batch size (modify `validation_reward_fn.py`)
2. Increase `test_freq` (validate less often)
3. Use CPU for validation (change `device` in `validation_reward_fn.py`)

### Speed Impact

Validation adds time to training:
- Training step: ~27 seconds (your current)
- Validation step: +10-30 seconds (depends on val set size)

With `test_freq=2`, validation runs every other step:
- Step 0: Train (27s)
- Step 1: Train (27s)
- Step 2: Train + Validate (27s + 15s = 42s)
- Step 3: Train (27s)
- ...

## Troubleshooting

### Issue: "No prompt found in batch data"

**Cause**: `data.return_raw_chat=True` not set

**Fix**: Add to training script:
```bash
data.return_raw_chat=True
```

### Issue: Out of GPU memory during validation

**Solution 1**: Reduce validation batch size

Edit `validation_reward_fn.py`:
```python
# In compute_score function, when tokenizing:
inputs = rm_tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=2048,  # Reduce from 4096
)
```

**Solution 2**: Use CPU for validation

Edit `validation_reward_fn.py`:
```python
# In _load_reward_model function:
_RM_DEVICE = torch.device("cpu")  # Force CPU
```

### Issue: Validation is slow

**Options**:
1. Validate less frequently: `trainer.test_freq=10`
2. Use smaller validation set (create a subset in preprocessing)
3. Skip validation and use post-training evaluation instead

## Comparison: Training vs Validation Rewards

### What to Expect

**Normal behavior:**
- Training rewards increase over time
- Validation rewards may increase slower
- Some gap between training and validation is normal (overfitting)

**Warning signs:**
- Training rewards increase but validation decreases → Overfitting
- Validation rewards don't improve at all → Model not generalizing
- Large gap (>2x) between train/val → Serious overfitting

### Example Interpretation

```
Step 10:
  critic/rewards/mean: 0.93  (training)
  test/reward/mean: 0.85      (validation)
  Gap: 0.08 → Normal

Step 50:
  critic/rewards/mean: 2.45   (training)
  test/reward/mean: 1.92      (validation)
  Gap: 0.53 → Slight overfitting, acceptable

Step 100:
  critic/rewards/mean: 4.12   (training)
  test/reward/mean: 1.45      (validation)
  Gap: 2.67 → Severe overfitting! Consider stopping
```

## Next Steps

1. ✅ **Train with validation enabled** - Use updated scripts
2. ✅ **Monitor W&B** - Watch `test/reward/mean`
3. ✅ **Compare train/val curves** - Check for overfitting
4. ✅ **Use post-training eval** - Your existing `run_full_evaluation.sh`

The combination of:
- Real-time validation during training
- Post-training rigorous evaluation

Gives you the most comprehensive view of model performance! 🎯

