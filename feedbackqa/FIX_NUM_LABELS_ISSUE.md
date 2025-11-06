# Fix: Model Has num_labels=2 After Training

## 🔴 Problem

You trained the model with the updated `rm_train.py`, but the saved model still has `num_labels=2` instead of `num_labels=1`.

```
✗ Expected num_labels=1, got 2
```

---

## 🤔 Why This Happened

The training script loaded from an **existing checkpoint** that had `num_labels=2`. When you resume training from a checkpoint, the model configuration comes from the checkpoint, not from the code.

**Timeline**:
1. You first trained with old script → saved model with `num_labels=2`
2. Script was updated → now has `num_labels=1` in code
3. You trained again → but training **resumed from checkpoint** with `num_labels=2`
4. Final model still has `num_labels=2` ❌

---

## ✅ Solution: Clean Retrain

You need to **completely delete** the old model and train from scratch.

### Option 1: Automated (Recommended)

```bash
bash feedbackqa/clean_retrain.sh
```

This script will:
- ✓ Delete old model completely
- ✓ Verify script has correct configuration
- ✓ Train from scratch with `num_labels=1`
- ✓ Verify the trained model
- ✓ Confirm BCE loss was used

### Option 2: Manual

```bash
# Step 1: Delete old model COMPLETELY
rm -rf ./feedback_qa_reward_model

# Step 2: Verify script is updated
grep "num_labels = 1" feedbackqa/rm_train.py
# Should show: config.num_labels = 1  ✅

# Step 3: Train from scratch
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-5

# Step 4: Verify
python feedbackqa/verify_bce_training.py
```

---

## 🔍 What to Look For During Training

### 1. At Start (Model Setup)

You should see:
```
Model and tokenizer setup complete
Model configuration: 1 labels, single_label_classification  ← Should be "1"!
Classification head dropout: 0.1
Loss function: Binary Cross-Entropy (BCE) - optimal for binary labels  ✅
```

**If you see "2 labels" here**, something is wrong - STOP and check the script!

### 2. During Training

```
Epoch 1/3:
  train_loss: 0.421  ← BCE loss values (0.2-0.6 range)
  val_accuracy: 0.78
```

### 3. After Training

Check the config:
```bash
cat ./feedback_qa_reward_model/final_model/config.json | grep num_labels
```

Should show:
```json
"num_labels": 1,  ✅
```

---

## ✅ Verification After Retraining

Run these checks:

### Check 1: Model Config
```bash
cat ./feedback_qa_reward_model/final_model/config.json | grep num_labels
```
**Expected**: `"num_labels": 1,` ✅

### Check 2: Training Summary
```bash
cat ./feedback_qa_reward_model/training_summary.json | grep loss_function
```
**Expected**: `"loss_function": "binary_cross_entropy",` ✅

### Check 3: Comprehensive Verification
```bash
python feedbackqa/verify_bce_training.py
```
**Expected**: 
```
✓ num_labels=1 (correct for verl)
✓ loss_function: binary_cross_entropy
✅ All checks passed!
```

### Check 4: verl Compatibility
```bash
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```
**Expected**: `✓ VERL COMPATIBILITY VERIFIED ✅`

---

## 🚫 Common Mistakes

### Mistake 1: Not Deleting Old Model

```bash
# ❌ WRONG - this will resume from checkpoint
python feedbackqa/rm_train.py ...
```

The training will load the old checkpoint and keep `num_labels=2`!

**Fix**: Delete first!
```bash
# ✅ CORRECT
rm -rf ./feedback_qa_reward_model
python feedbackqa/rm_train.py ...
```

### Mistake 2: Deleting Only final_model/

```bash
# ❌ WRONG - checkpoints still exist
rm -rf ./feedback_qa_reward_model/final_model
```

Other checkpoints in `./feedback_qa_reward_model/` can still be loaded!

**Fix**: Delete entire directory!
```bash
# ✅ CORRECT
rm -rf ./feedback_qa_reward_model
```

### Mistake 3: Not Checking During Training

If you don't check the logs at the start, you might train for hours only to find it's still using `num_labels=2`!

**Fix**: Always check the first few lines of training output!

---

## 📊 Expected Timeline

```
Clean Retrain Process:
  1. Delete old model         → 1 second
  2. Load base model          → 30 seconds
  3. Train (3 epochs)         → 10-20 minutes
  4. Save model               → 1 minute
  5. Verify                   → 10 seconds
  
Total: ~15-25 minutes
```

---

## 🎯 Quick Command Reference

```bash
# Complete clean retrain (one command)
bash feedbackqa/clean_retrain.sh

# Or manual steps
rm -rf ./feedback_qa_reward_model && \
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct && \
python feedbackqa/verify_bce_training.py
```

---

## ✅ Success Criteria

After clean retrain, you should have:

- [x] `config.json` shows `"num_labels": 1`
- [x] `training_summary.json` shows `"loss_function": "binary_cross_entropy"`
- [x] Training logs showed "Loss function: Binary Cross-Entropy (BCE)"
- [x] `verify_bce_training.py` passes all checks
- [x] `verify_trained_model.py` confirms verl compatibility
- [x] Model architecture is `LlamaForTokenClassification`

---

## 🆘 Still Having Issues?

### Issue: Script still loads checkpoint

**Check**: Are there checkpoint files?
```bash
ls -la ./feedback_qa_reward_model/
```

**If you see** `checkpoint-*` directories:
```bash
# Nuclear option - delete EVERYTHING
rm -rf ./feedback_qa_reward_model
# Make sure directory is gone
ls ./feedback_qa_reward_model  # Should say "No such file or directory"
```

### Issue: Training says "num_labels: 2" at start

**This means**: The script is not using the updated code!

**Check**:
```bash
grep -n "config.num_labels" feedbackqa/rm_train.py
```

**Should show**:
```
319:        config.num_labels = 1  # Single scalar reward per token (verl-compatible)
```

**If not**: Make sure you saved the file after my updates!

### Issue: Model loads but has wrong architecture

**Symptom**: Model type is `LlamaForSequenceClassification` instead of `LlamaForTokenClassification`

**This means**: You're loading an old model!

**Fix**:
```bash
# Make absolutely sure old model is gone
rm -rf ./feedback_qa_reward_model
rm -rf ~/.cache/huggingface/hub/models--*feedback*  # Clear cache if needed
# Then retrain
```

---

## 📝 Summary

**Problem**: Old checkpoint with `num_labels=2` is being loaded
**Solution**: Delete old model completely and retrain from scratch
**Command**: `bash feedbackqa/clean_retrain.sh` or manual deletion + retrain
**Verification**: Run `verify_bce_training.py` - should show `num_labels=1` ✅

---

**TL;DR**: 
```bash
rm -rf ./feedback_qa_reward_model
python feedbackqa/rm_train.py --train_file ... --model_name ...
python feedbackqa/verify_bce_training.py
```

Look for `num_labels: 1` in the verification output! ✅

