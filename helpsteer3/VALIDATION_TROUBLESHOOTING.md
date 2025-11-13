# Validation Metrics Troubleshooting Guide

## Problem
Validation metrics are not appearing in wandb logs during PPO training with reward models.

## Root Causes

### 1. **Validation Not Running**
- Check if `trainer.test_freq` is set (e.g., `trainer.test_freq=2`)
- Set `trainer.val_before_train=True` to ensure validation setup is correct

### 2. **Reward Model Not Evaluated During Validation**
The key issue is that verl's validation loop may not automatically compute reward model scores during validation. This is because:
- Training uses the reward model to score generated responses
- Validation needs to be explicitly configured to do the same
- The validation loop might skip reward computation by default for efficiency

## Solutions

### Solution 1: Enable `val_before_train`
```bash
trainer.val_before_train=True  # Changed from False
```

This ensures:
- Validation runs at least once before training starts
- You can verify validation is working correctly
- Reward model is properly loaded for validation

### Solution 2: Check Validation Data Path
Make sure your validation data is correctly specified:
```bash
data.val_files="$DATA_DIR/validation.parquet"
```

### Solution 3: Monitor Console Output
During validation, you should see logs like:
```
[INFO] Running validation at step X
[INFO] Validation: Processing batch Y/Z
[INFO] Validation metrics: {...}
```

If you don't see these, validation might not be running.

### Solution 4: Use Lower `test_freq` for More Frequent Validation
```bash
trainer.test_freq=1  # Validate every step (slower but more visibility)
# or
trainer.test_freq=5  # Validate every 5 steps
```

## What Validation Metrics Should You See?

When working correctly, you should see metrics prefixed with `val/` in wandb:
- `val/reward/mean` - Mean reward score on validation set
- `val/reward/max` - Max reward score
- `val/reward/min` - Min reward score
- `val/response_length/mean` - Average response length
- `val/prompt_length/mean` - Average prompt length

## Logging Responses to Wandb

To log actual responses (not just metrics) to wandb, you need custom logging. See the script below.

### Option 1: Post-hoc Logging (Recommended)
After training, extract and log responses:

```python
# See log_responses_to_wandb.py script
```

### Option 2: During Training (Advanced)
Modify the trainer to log responses during validation. This requires:
1. Custom callback in verl trainer
2. Access to generated responses during validation
3. Logging to wandb tables

**Note**: This is not straightforward in verl and may require modifying verl source code.

## Debugging Steps

### Step 1: Check if Validation is Running
Look for these in your logs:
```bash
grep "validation\|val_" your_training_log.txt
```

### Step 2: Verify Validation Data is Loaded
```bash
# Should see: "Loaded X validation examples"
grep "validation.*parquet\|val_files" your_training_log.txt
```

### Step 3: Check Reward Model Logs
```bash
# Should see reward model being used during validation
grep "reward.*val\|Validation.*reward" your_training_log.txt
```

### Step 4: Inspect Wandb Directly
1. Go to your wandb project: `helpsteer3_ppo_experiment`
2. Click on your run
3. Go to "Charts" tab
4. Search for metrics starting with `val/`
5. If no `val/` metrics exist, validation is not computing rewards

## Common Issues

### Issue 1: "No validation metrics appear"
**Cause**: Validation runs but doesn't compute rewards
**Fix**: This is a limitation of verl when using reward models. The validation loop may only check basic metrics (response length) but not call the reward model.

**Workaround**: Run manual validation after training using the evaluation script:
```bash
python helpsteer3/inference.py --model_path <checkpoint> --output_file val_predictions.json
python helpsteer3/evaluate.py --predictions_file val_predictions.json --output_file val_evaluation.json
```

### Issue 2: "OOM during validation"
**Cause**: Validation batch size too large
**Fix**: Validation uses the same batch size as training, which might be too large if you have limited memory.

### Issue 3: "Validation is too slow"
**Cause**: Reward model evaluation is expensive
**Fix**: 
- Increase `test_freq` to validate less frequently
- Use a smaller validation set
- Reduce validation batch size

## Recommended Configuration

For best visibility during training:

```bash
# Enable validation before training (verify setup)
trainer.val_before_train=True

# Validate frequently (every 2 steps) but not too often
trainer.test_freq=2

# Ensure validation data is specified
data.val_files="$DATA_DIR/validation.parquet"

# Log to both console and wandb
trainer.logger='["console","wandb"]'
```

## Alternative: Manual Validation

If automatic validation doesn't work well, run manual validation periodically:

1. **During training**: Save checkpoints frequently
```bash
trainer.save_freq=5  # Save every 5 steps
```

2. **After training**: Run evaluation on saved checkpoints
```bash
# For each checkpoint
bash helpsteer3/run_full_evaluation.sh
```

This gives you:
- Complete control over validation
- Detailed metrics and responses
- Statistical analysis
- Response samples logged to files

## Summary

The core issue is that verl's validation with reward models may not automatically compute and log reward scores. The best approaches are:

1. ✅ **Set `val_before_train=True`** - Ensures validation setup is correct
2. ✅ **Monitor console output** - Check if validation actually runs
3. ✅ **Use manual evaluation** - Most reliable for detailed validation metrics
4. ⚠️ **Custom logging** - Requires modifying verl source (advanced)

For logging responses to wandb, see `log_responses_to_wandb.py` script.

