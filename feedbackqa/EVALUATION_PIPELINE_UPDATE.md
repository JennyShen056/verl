# Evaluation Pipeline Update for num_labels=1 Reward Models

## Summary

The evaluation pipeline has been updated to support both `num_labels=1` (scalar regression) and `num_labels=2` (binary classification) reward models, with proper detection and handling of each format.

## Updated Files

### 1. `rm_train.py` ✅
**Changes**: Converted from binary classification to scalar regression
- `num_labels=2` → `num_labels=1`
- `problem_type="single_label_classification"` → `problem_type="regression"`
- Binary labels (0, 1) → Continuous targets (-1.0, +1.0)
- Output: Single scalar reward score

**Status**: ✅ Updated and ready to use

### 2. `evaluate.py` ✅
**Changes**: Improved reward score extraction with format detection

**Key improvements**:
- Automatically detects model format from `logits.shape[-1]`
- Handles `num_labels=1` (scalar regression) correctly
- Handles `num_labels=2` (binary classification) for backward compatibility
- Shows format detection in output

**Before**:
```python
if outputs.logits.shape[-1] == 2:
    scores = outputs.logits[:, 1].cpu().numpy()
else:
    scores = outputs.logits.squeeze(-1).cpu().numpy()
```

**After**:
```python
logits = outputs.logits

if logits.shape[-1] == 1:
    # Scalar regression (num_labels=1): Direct reward score
    # This is the correct format for verl-compatible reward models
    scores = logits.squeeze(-1).cpu().numpy()
elif logits.shape[-1] == 2:
    # Binary classification (num_labels=2): Use positive class logit
    # Note: This format is NOT compatible with verl PPO training
    scores = logits[:, 1].cpu().numpy()
else:
    raise ValueError(f"Unexpected logits shape: {logits.shape}")
```

**Status**: ✅ Updated and backward compatible

### 3. `verify_trained_model.py` ✅
**Changes**: Complete rewrite to test `num_labels=1` models correctly

**Major fixes**:
- ❌ Was using `AutoModelForTokenClassification` (WRONG!)
- ✅ Now uses `AutoModelForSequenceClassification` (CORRECT!)
- ❌ Was checking for token-level outputs `[batch, seq_len, 1]`
- ✅ Now checks for sequence-level outputs `[batch, 1]`
- Added batch inference testing
- Added comprehensive model summary

**What it verifies**:
1. ✅ `config.num_labels == 1`
2. ✅ `config.problem_type == "regression"`
3. ✅ Model loads with `AutoModelForSequenceClassification`
4. ✅ Output shape is `[batch_size, 1]`
5. ✅ Can extract scalar reward scores
6. ✅ Batch inference works correctly

**Status**: ✅ Fixed and comprehensive

### 4. `inference.py` ✅
**Status**: No changes needed - handles policy model (generation), not reward model

### 5. `compare_results.py` ✅
**Status**: No changes needed - compares reward scores agnostic of model format

### 6. `create_test_subset.py` ✅
**Status**: No changes needed - just creates data subsets

### 7. `run_full_evaluation.sh` ✅
**Status**: No changes needed - orchestrates the pipeline

## Usage

### 1. Train New Reward Model (num_labels=1)

```bash
cd feedbackqa
python rm_train.py \
    --train_file feedback_train_rm.json \
    --valid_file feedback_valid_rm.json \
    --test_file feedback_test_rm.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./feedback_qa_reward_model_v2 \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5
```

### 2. Verify Model Format

```bash
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model_v2/final_model
```

**Expected output**:
```
✅ SUCCESS! Your model is verl-compatible!
======================================================================

📋 Model Summary:
  • Architecture: AutoModelForSequenceClassification
  • num_labels: 1 (scalar regression)
  • problem_type: regression
  • Output shape: [batch_size, 1]
  • Output range: unbounded (-∞, +∞)

🚀 You can now use this model with verl PPO training:
  reward_model.enable=True \
  reward_model.model.path=./feedback_qa_reward_model_v2/final_model
```

### 3. Run Full Evaluation Pipeline

```bash
bash feedbackqa/run_full_evaluation.sh
```

This will:
1. Merge checkpoints from PPO training
2. Run inference on test set
3. Evaluate using reward model (automatically detects format)
4. Generate comparison statistics

### 4. Evaluate with Your Custom Reward Model

```bash
python feedbackqa/evaluate.py \
    --predictions_file outputs/case1_predictions.json \
    --reward_model_path ./feedback_qa_reward_model_v2/final_model \
    --output_file outputs/case1_evaluation.json \
    --batch_size 8
```

**Expected output**:
```
Loading reward model from: ./feedback_qa_reward_model_v2/final_model
✓ Reward model loaded
  Type: LlamaForSequenceClassification
  num_labels: 1
  Format: Scalar regression (verl-compatible)
  Device: cuda:0
```

## Backward Compatibility

The evaluation pipeline maintains **backward compatibility** with old `num_labels=2` models:

### Old Models (num_labels=2)
- ⚠️ **NOT compatible with verl PPO training**
- ✅ **CAN still be used for evaluation**
- Uses positive class logit (index 1) as score
- Will display: `Format: Binary classification (NOT verl-compatible)`

### New Models (num_labels=1)
- ✅ **Compatible with verl PPO training**
- ✅ **Compatible with evaluation pipeline**
- Uses direct scalar output as score
- Will display: `Format: Scalar regression (verl-compatible)`

## Model Format Comparison

| Aspect | Old (num_labels=2) | New (num_labels=1) |
|--------|-------------------|-------------------|
| **Training approach** | Binary classification | Scalar regression |
| **Loss function** | Cross-entropy | MSE |
| **Output shape** | `[batch, 2]` | `[batch, 1]` |
| **Output values** | 2 logits | 1 scalar |
| **Label format** | `{0, 1}` | `{-1.0, 1.0}` |
| **verl Compatible** | ❌ No | ✅ Yes |
| **Evaluation Compatible** | ✅ Yes | ✅ Yes |
| **Prediction threshold** | Softmax > 0.5 | Score > 0.0 |

## Troubleshooting

### Issue: Model verification fails with "Expected num_labels=1"

**Cause**: You're testing an old model trained with `num_labels=2`

**Solution**: Retrain with the updated `rm_train.py`

### Issue: "Expected shape [batch, 1], got [batch, 2]"

**Cause**: Model was trained with binary classification

**Solution**: Retrain with scalar regression format

### Issue: verl PPO training fails with shape mismatch

**Cause**: Using a `num_labels=2` model with verl

**Solution**: 
1. Verify your model with `verify_trained_model.py`
2. If it fails, retrain with the updated `rm_train.py`
3. Verify again before using with verl

### Issue: Evaluation works but verl PPO fails

**Cause**: The evaluation pipeline accepts both formats, but verl only accepts `num_labels=1`

**Solution**: Always verify models with `verify_trained_model.py` before using with verl

## Testing Checklist

Before using a reward model with verl PPO:

- [ ] Run `verify_trained_model.py` - should pass all checks
- [ ] Check that `num_labels=1` in config
- [ ] Check that `problem_type="regression"` in config
- [ ] Verify output shape is `[batch_size, 1]`
- [ ] Test evaluation pipeline with `evaluate.py`
- [ ] Confirm format shows "Scalar regression (verl-compatible)"

## Files Summary

| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `rm_train.py` | Train RM | ✅ Updated | Now produces num_labels=1 |
| `evaluate.py` | Score predictions | ✅ Updated | Detects format automatically |
| `verify_trained_model.py` | Verify compatibility | ✅ Fixed | Now tests correctly |
| `inference.py` | Generate predictions | ✅ No change | Policy model only |
| `compare_results.py` | Compare experiments | ✅ No change | Format agnostic |
| `create_test_subset.py` | Create test subset | ✅ No change | Data processing only |
| `run_full_evaluation.sh` | Run pipeline | ✅ No change | Orchestration |

## Migration Path

If you have an old reward model:

1. **Keep old model for reference**:
   ```bash
   mv feedback_qa_reward_model feedback_qa_reward_model_old_v2
   ```

2. **Train new model**:
   ```bash
   python rm_train.py [args] --output_dir feedback_qa_reward_model_new
   ```

3. **Verify new model**:
   ```bash
   python verify_trained_model.py --model_path feedback_qa_reward_model_new/final_model
   ```

4. **Compare evaluation results** (optional):
   ```bash
   # Evaluate with old model
   python evaluate.py --reward_model_path feedback_qa_reward_model_old_v2/final_model \
       --predictions_file outputs/predictions.json \
       --output_file outputs/eval_old.json
   
   # Evaluate with new model
   python evaluate.py --reward_model_path feedback_qa_reward_model_new/final_model \
       --predictions_file outputs/predictions.json \
       --output_file outputs/eval_new.json
   ```

## References

- `REWARD_MODEL_FORMAT_EXPLANATION.md` - Detailed format explanation
- RLHFlow Bradley-Terry: https://github.com/RLHFlow/RLHF-Reward-Modeling
- Working example: `sfairXC/FsfairX-LLaMA3-RM-v0.1`

## Questions?

See the comprehensive explanation in:
- `feedbackqa/REWARD_MODEL_FORMAT_EXPLANATION.md`
- `feedbackqa/QUESTION_ONLY_RM_GUIDE.md`

