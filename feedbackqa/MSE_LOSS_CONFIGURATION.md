# MSE Loss Configuration for Reward Model

## ✅ Configuration Changed to MSE Loss

The reward model training has been updated to use **Mean Squared Error (MSE)** loss instead of Binary Cross-Entropy (BCE) loss.

---

## 🔧 What Changed

### 1. Loss Function

**Old (BCE)**:
```python
loss = F.binary_cross_entropy_with_logits(
    eos_logits, 
    labels.float(),
    reduction='mean'
)
```

**New (MSE)** ✅:
```python
loss = F.mse_loss(
    eos_logits, 
    labels.float(),
    reduction='mean'
)
```

### 2. Training Logs

**Will show**:
```
Loss function: Mean Squared Error (MSE) - regression-based approach
```

### 3. Training Summary

**Will contain**:
```json
{
  "loss_function": "mean_squared_error"
}
```

---

## 📊 What MSE Does

### Loss Computation

For a single example:
```
Prediction (logit): 0.85
True label: 1.0

MSE Loss = (0.85 - 1.0)² = 0.0225
```

### Characteristics

| Aspect | MSE Loss |
|--------|----------|
| **Type** | Regression loss |
| **Penalty** | Quadratic (squared error) |
| **Gradient** | Linear: 2(prediction - label) |
| **Output** | Raw logits (any real number) |
| **For inference** | Apply sigmoid to get probabilities |

---

## 🎯 Expected Training Behavior

### Loss Values

With MSE, expect different loss ranges than BCE:

```
Epoch 1: train_loss=0.145, val_accuracy=0.78
Epoch 2: train_loss=0.098, val_accuracy=0.85
Epoch 3: train_loss=0.067, val_accuracy=0.88
```

**Typical MSE loss range**: 0.05 - 0.25
- Much lower than BCE (which is typically 0.2 - 0.6)
- Different scale, but both work fine!

### Gradients

MSE gradients:
```
When prediction is 0.1 and label is 1.0:
  Gradient = 2(0.1 - 1.0) = -1.8

When prediction is 0.9 and label is 1.0:
  Gradient = 2(0.9 - 1.0) = -0.2
```

---

## ✅ Still verl-Compatible

**Important**: The model is still fully compatible with verl!

- ✅ `num_labels=1`
- ✅ `AutoModelForTokenClassification`
- ✅ Output shape: `[batch_size, seq_len, 1]`
- ✅ EOS token scoring
- ✅ Can be used for PPO training

**The loss function only affects training, not the model architecture or inference!**

---

## 🔍 How to Verify

### Check Training Logs

Look for at start:
```
Loss function: Mean Squared Error (MSE) - regression-based approach  ✅
```

### Check Training Summary

```bash
cat ./feedback_qa_reward_model/training_summary.json | grep loss_function
```

Should show:
```json
"loss_function": "mean_squared_error",  ✅
```

### Run Verification

```bash
python feedbackqa/verify_bce_training.py
```

**Note**: The script name says "bce" but it works for MSE too! It just verifies the model structure.

Expected output:
```
✓ num_labels=1 (correct for verl)
✓ Model predictions working
✅ All checks passed! Model is working correctly.
```

---

## 🚀 Training Commands

### Clean Retrain with MSE

```bash
# Delete old model
rm -rf ./feedback_qa_reward_model

# Train with MSE (automatic with updated code)
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 4 \
    --num_epochs 3 \
    --learning_rate 2e-5
```

### Verify

```bash
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```

---

## 📈 Comparison: MSE vs BCE

| Aspect | MSE | BCE |
|--------|-----|-----|
| **Loss type** | Regression | Classification |
| **Loss range** | 0.05 - 0.25 | 0.2 - 0.6 |
| **Gradient when very wrong** | -1.8 | -99.0 |
| **Gradient when close** | -0.2 | -0.1 |
| **Theoretically correct for binary?** | No | Yes |
| **Works in practice?** | Yes | Yes |
| **verl compatible?** | Yes ✅ | Yes ✅ |

**Both work!** Choose based on your experimental needs.

---

## 💡 When to Use MSE vs BCE

### Use MSE if:
- You want regression-based training
- You prefer symmetric penalty
- You're comparing with other MSE-based models
- You want simpler loss computation

### Use BCE if:
- You want theoretically correct binary classification
- You want stronger gradients when very wrong
- You want proper probability calibration
- You're following best practices for binary tasks

---

## 🔄 How to Switch Back to BCE

If you want to switch back to BCE:

1. **Change the loss function**:
```python
# In RewardModelTrainer.compute_loss()
loss = F.binary_cross_entropy_with_logits(
    eos_logits, 
    labels.float(),
    reduction='mean'
)
```

2. **Update the log message**:
```python
self.logger.info("Loss function: Binary Cross-Entropy (BCE) - optimal for binary labels")
```

3. **Update the summary**:
```python
"loss_function": "binary_cross_entropy",
```

4. **Retrain** from scratch

---

## ✅ Ready to Train with MSE

Everything is configured for MSE loss. Just run:

```bash
# Clean retrain
rm -rf ./feedback_qa_reward_model

# Train
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct

# Verify
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```

**Training will use MSE loss!** ✅

---

## 📝 Summary

- ✅ **Loss function**: Changed to MSE
- ✅ **Model architecture**: Still `num_labels=1` (verl-compatible)
- ✅ **Training**: Will show "Loss function: Mean Squared Error (MSE)"
- ✅ **Loss values**: Expect 0.05 - 0.25 range (lower than BCE)
- ✅ **Verification**: Same scripts work for both MSE and BCE
- ✅ **PPO training**: Model works the same way for verl

**Ready to train!** 🚀

