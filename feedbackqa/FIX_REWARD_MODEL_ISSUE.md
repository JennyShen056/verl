# 🔴 URGENT: Fix Reward Model Issue

## Problem

Your reward model has `num_labels=2`, but verl requires `num_labels=1`.

```bash
❌ VERIFICATION FAILED: Expected num_labels=1, got 2
```

---

## Quick Fix (3 Steps)

### 1. Delete Old Model

```bash
rm -rf ./feedback_qa_reward_model
```

### 2. Retrain with Updated Script

The `rm_train.py` script has been updated to be verl-compatible. Run:

```bash
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 2e-5
```

**Expected time**: 10-20 minutes on single A100

### 3. Verify It Works

```bash
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```

**Expected output**:
```
✓ Loading model config...
  num_labels: 1  ✅

✓ Model architecture verified
  Using: AutoModelForTokenClassification

✓ Output shape verified
  Batch output shape: [1, sequence_length, 1]

✓ VERL COMPATIBILITY VERIFIED ✅
```

---

## What Changed in `rm_train.py`?

| Component | Old | New |
|-----------|-----|-----|
| Model class | `AutoModelForSequenceClassification` | `AutoModelForTokenClassification` ✅ |
| `num_labels` | 2 | 1 ✅ |
| Label type | `int` | `float` ✅ |
| Output shape | `[batch, 2]` | `[batch, seq_len, 1]` ✅ |

---

## Data Format During Training

Your data is formatted as conversational Q&A pairs:

### Input Data (`feedback_train_rm.json`)

```json
{
  "question": "How do I get help finding a job?",
  "section_content": "In this rapidly changing jobs market...",
  "label": 1
}
```

### After Chat Template Formatting

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

How do I get help finding a job?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

In this rapidly changing jobs market...<|eot_id|>
```

### After Tokenization

```python
{
  "input_ids": [128000, 128006, 882, ..., 128009],
  "attention_mask": [1, 1, 1, ..., 1],
  "labels": 1.0  # Float label for verl compatibility
}
```

### Model Output

```
Input:  [batch_size, seq_len] token IDs
Output: [batch_size, seq_len, 1] scores (one per token)

Prediction: Take last token (EOS) score → apply sigmoid → threshold at 0.5
```

---

## Why This Matters

**verl PPO training** expects the reward model to output:
- Single scalar reward per sequence
- From token classification architecture
- With `num_labels=1`

Your old model with `num_labels=2` uses:
- Two-class classification (good/bad)
- Sequence classification architecture
- Not compatible with verl's expectations

---

## Training Metrics to Expect

```
Epoch 1/3:
  train_loss: 0.245
  val_accuracy: 0.785
  val_f1: 0.812

Epoch 2/3:
  train_loss: 0.189
  val_accuracy: 0.845
  val_f1: 0.867

Epoch 3/3:
  train_loss: 0.152
  val_accuracy: 0.872  ✅ Good!
  val_f1: 0.891  ✅ Good!
  val_auc: 0.925  ✅ Good!
```

**Target metrics**:
- Accuracy > 0.85
- F1 > 0.85
- AUC > 0.90

---

## After Retraining

Once the model is retrained and verified:

1. ✅ Reward model is verl-compatible
2. ⏭️ Preprocess data for PPO training:
   ```bash
   python feedbackqa/preprocess_ppo_case1_question_only.py
   python feedbackqa/preprocess_ppo_case2_with_feedback.py
   ```

3. ⏭️ Run PPO training:
   ```bash
   bash feedbackqa/run_ppo_case1_question_only.sh
   bash feedbackqa/run_ppo_case2_with_feedback.sh
   ```

4. ⏭️ Compare results in W&B!

---

## Need More Details?

- **Data format explanation**: `RM_DATA_FORMAT_AND_RETRAINING.md`
- **Full retraining guide**: `README_VERL_INTEGRATION.md`
- **Verification tool**: `verify_trained_model.py`

---

## Still Having Issues?

### Issue: OOM during training

**Solution**: Reduce batch size
```bash
python feedbackqa/rm_train.py ... --batch_size 2
```

### Issue: Low accuracy (<0.7)

**Possible causes**:
- Not enough training data
- Imbalanced classes (too many positive or negative examples)
- Learning rate too high

**Solution**: Train for more epochs or adjust learning rate
```bash
python feedbackqa/rm_train.py ... --epochs 5 --learning_rate 1e-5
```

### Issue: Model verification still fails

**Check**:
1. Did you delete the old model folder?
2. Did you run the updated `rm_train.py`?
3. Check `./feedback_qa_reward_model/final_model/config.json` should have `"num_labels": 1`

---

**Quick Summary**: Delete old model → Retrain with updated script → Verify → Continue with PPO training 🚀

