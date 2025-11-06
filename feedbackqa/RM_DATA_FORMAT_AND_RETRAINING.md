# Reward Model: Data Format & Retraining Guide

## 🔴 Issue: Model Has `num_labels=2` Instead of `num_labels=1`

**Problem**: Your trained model has `num_labels=2`, but verl requires `num_labels=1`.

**Root Cause**: The model was trained with an older version of `rm_train.py` that used binary classification (`AutoModelForSequenceClassification` with 2 labels) instead of token classification.

**Solution**: Retrain the model with the updated `rm_train.py` script.

---

## ✅ How to Fix: Retrain the Reward Model

### Step 1: Verify the Updated Script

The `rm_train.py` has been updated to be verl-compatible. Key changes:

```python
# OLD (Binary Classification - NOT verl-compatible)
from transformers import AutoModelForSequenceClassification
config.num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(...)

# NEW (Token Classification - verl-compatible) ✅
from transformers import AutoModelForTokenClassification
config.num_labels = 1
model = AutoModelForTokenClassification.from_pretrained(...)
```

### Step 2: Delete Old Model

```bash
# Remove the old model to avoid confusion
rm -rf ./feedback_qa_reward_model
```

### Step 3: Retrain with Updated Script

```bash
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --batch_size 4 \
    --epochs 3 \
    --learning_rate 2e-5 \
    --max_length 1024
```

### Step 4: Verify the New Model

```bash
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```

**Expected Output**:
```
✓ Loading model config...
  num_labels: 1  ✅

✓ Model architecture verified
  Using: AutoModelForTokenClassification

✓ VERL COMPATIBILITY VERIFIED ✅
```

---

## 📊 Data Format During Training

### Input Data Format (`feedback_train_rm.json`)

The raw training data is in JSON format with question-answer pairs and binary labels:

```json
[
  {
    "question": "How do I get help finding a job?",
    "section_content": "In this rapidly changing jobs market, it can be difficult to know where to start when looking for employment. The good news is that there are many resources available...",
    "label": 1
  },
  {
    "question": "What are common interview mistakes?",
    "section_content": "Some mistakes include being late, not dressing appropriately...",
    "label": 0
  }
]
```

**Fields**:
- `question`: The user's question
- `section_content`: The answer/response
- `label`: Binary label
  - `1` = Good answer (positive example)
  - `0` = Bad answer (negative example)

### Data Transformation Pipeline

#### Step 1: Format with Chat Template

Each example is formatted using the model's chat template:

```python
def format_with_chat_template(question, section_content):
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": section_content}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return text
```

**Example Output** (for Llama-3.1):
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

How do I get help finding a job?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

In this rapidly changing jobs market, it can be difficult to know where to start when looking for employment. The good news is that there are many resources available...<|eot_id|>
```

#### Step 2: Tokenization

```python
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=1024,
        return_tensors=None,
    )
    # Convert binary labels (0, 1) to float scores
    tokenized["labels"] = [float(label) for label in examples["label"]]
    return tokenized
```

**Example**:
```python
Input text: "<|begin_of_text|>...<|eot_id|>"
Label: 1

After tokenization:
{
    "input_ids": [128000, 128006, 882, ...],  # Token IDs
    "attention_mask": [1, 1, 1, ...],         # Attention mask
    "labels": 1.0                              # Float label (for verl)
}
```

#### Step 3: Model Training

**OLD Format (Binary Classification)**:
```
Input: [batch_size, seq_len] token IDs
Model: AutoModelForSequenceClassification
Output: [batch_size, 2] logits (class 0 and class 1)
Loss: Cross-entropy between output and integer label
```

**NEW Format (verl-compatible Token Classification)** ✅:
```
Input: [batch_size, seq_len] token IDs
Model: AutoModelForTokenClassification
Output: [batch_size, seq_len, 1] logits (one score per token)
Prediction: Take last token (EOS) score
Loss: MSE between last token score and float label
```

### Complete Data Flow Example

**Raw Data**:
```json
{
  "question": "How do I get help finding a job?",
  "section_content": "In this rapidly changing jobs market...",
  "label": 1
}
```

**↓ Step 1: Format with Chat Template**
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

How do I get help finding a job?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

In this rapidly changing jobs market...<|eot_id|>
```

**↓ Step 2: Tokenize**
```python
{
  "input_ids": [128000, 128006, 882, ..., 128009],  # 287 tokens
  "attention_mask": [1, 1, 1, ..., 1],
  "labels": 1.0  # Float, not int!
}
```

**↓ Step 3: Model Forward Pass**
```python
Input shape: [4, 287]  # batch_size=4, seq_len=287
Output shape: [4, 287, 1]  # One score per token
```

**↓ Step 4: Extract EOS Token Score**
```python
# Take the last token's score (EOS token)
eos_scores = output[:, -1, 0]  # Shape: [4]
# Values: [0.85, 0.12, 0.93, 0.45]
```

**↓ Step 5: Compute Loss**
```python
# MSE loss between predicted scores and true labels
loss = MSE(eos_scores, labels)
```

**↓ Step 6: Evaluation Metrics**
```python
# Apply sigmoid to get probabilities
probs = sigmoid(eos_scores)  # [0.7, 0.53, 0.72, 0.61]

# Threshold at 0.5 for binary classification
predictions = (probs > 0.5)  # [1, 1, 1, 1]
labels = [1, 0, 1, 1]

# Calculate metrics
accuracy = 0.75  # 3/4 correct
precision, recall, f1, auc...
```

---

## 🔍 Key Differences: Old vs New

| Aspect | OLD (Binary Classification) | NEW (verl-compatible) |
|--------|----------------------------|----------------------|
| **Model Type** | `AutoModelForSequenceClassification` | `AutoModelForTokenClassification` ✅ |
| **`num_labels`** | 2 (class 0 and class 1) | 1 (single scalar reward) ✅ |
| **Output Shape** | `[batch, 2]` | `[batch, seq_len, 1]` ✅ |
| **Label Type** | `int` (0 or 1) | `float` (0.0 or 1.0) ✅ |
| **Loss Function** | Cross-entropy | MSE (implicit) ✅ |
| **Prediction** | `argmax(logits)` | `sigmoid(last_token_logit)` ✅ |
| **verl Compatible** | ❌ NO | ✅ YES |

---

## 🧪 How to Verify Data Format

### Check Raw Data

```bash
# Peek at the first example
python3 << EOF
import json
with open('feedbackqa/feedback_train_rm.json') as f:
    data = json.load(f)
print(f"Total examples: {len(data)}")
print(f"\nFirst example:")
print(f"  Question: {data[0]['question'][:50]}...")
print(f"  Answer: {data[0]['section_content'][:50]}...")
print(f"  Label: {data[0]['label']}")
print(f"\nLabel distribution:")
print(f"  Positive (1): {sum(1 for d in data if d['label'] == 1)}")
print(f"  Negative (0): {sum(1 for d in data if d['label'] == 0)}")
EOF
```

### Check Formatted Data During Training

The training script logs examples:

```
Data Format Example:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Text: <|begin_of_text|><|start_header_id|>user<|end_header_id|>

How do I get help finding a job?<|eot_id|>...

Label: 1.0 (Good Answer)
Length: 287 tokens
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Check Tokenized Data

The script also shows statistics:

```
Dataset Statistics:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train set:      1,000 examples
Validation set:   200 examples
Test set:         200 examples

Token Length Statistics:
  Mean: 287.3 tokens
  Median: 245 tokens
  95th percentile: 523 tokens
  Max: 1,024 tokens

Label Distribution:
  Positive (1.0): 687 (68.7%)
  Negative (0.0): 313 (31.3%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🎯 Training Process Summary

### What Happens During Training

1. **Load Data**: Read JSON files with Q&A pairs and labels

2. **Format**: Apply chat template to create conversational format
   ```
   User: [question]
   Assistant: [answer]
   ```

3. **Tokenize**: Convert text to token IDs (max 1024 tokens)

4. **Train Model**: 
   - Model outputs one score per token `[batch, seq_len, 1]`
   - Use EOS token score as final prediction
   - Compare to float label (0.0 or 1.0)

5. **Evaluate**: 
   - Apply sigmoid to get probabilities
   - Threshold at 0.5 for binary classification
   - Calculate accuracy, precision, recall, F1, AUC

6. **Save Model**: Store as `AutoModelForTokenClassification` with `num_labels=1`

### Training Metrics to Watch

```
Epoch 1/3:
  Training loss: 0.245
  Validation accuracy: 0.785
  Validation F1: 0.812
  Validation AUC: 0.856

Epoch 2/3:
  Training loss: 0.189
  Validation accuracy: 0.845
  Validation F1: 0.867
  Validation AUC: 0.903

Epoch 3/3:
  Training loss: 0.152
  Validation accuracy: 0.872  ✅
  Validation F1: 0.891  ✅
  Validation AUC: 0.925  ✅
```

**Good metrics**:
- Accuracy > 0.85
- F1 > 0.85
- AUC > 0.90

---

## 🔧 Troubleshooting

### Issue: Label Type Mismatch

**Error**: `RuntimeError: expected scalar type Float but found Int`

**Cause**: Labels are still integers instead of floats

**Fix**: Verify `tokenize_function` converts labels to float:
```python
tokenized["labels"] = [float(label) for label in examples["label"]]
```

### Issue: Shape Mismatch in Metrics

**Error**: `IndexError: index 1 is out of bounds for dimension 1 with size 1`

**Cause**: Trying to access `predictions[:, 1]` when `num_labels=1`

**Fix**: Updated `compute_metrics` handles this:
```python
if len(predictions.shape) == 3:
    probs = torch.sigmoid(torch.from_numpy(predictions[:, -1, 0])).numpy()
```

### Issue: Model Not Loading in verl

**Error**: `Model expected num_labels=1, got 2`

**Cause**: Old model still has 2 labels

**Fix**: Retrain model with updated script (see Step 2 above)

---

## 📝 Quick Reference

### Retrain Command

```bash
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct
```

### Verify Command

```bash
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```

### Expected Training Time

- **Hardware**: Single A100 GPU
- **Dataset**: ~1,400 examples
- **Time**: ~10-20 minutes for 3 epochs

---

## ✅ Success Checklist

After retraining:

- [ ] Model verification passes (`num_labels=1`)
- [ ] Model type is `AutoModelForTokenClassification`
- [ ] Training accuracy > 0.85
- [ ] Validation F1 > 0.85
- [ ] Model saved to `./feedback_qa_reward_model/final_model/`
- [ ] `training_summary.json` shows `verl_compatible: true`

---

## 🎓 Key Takeaways

1. **verl requires `num_labels=1`** with `AutoModelForTokenClassification`

2. **Data format is conversational**: User question + Assistant answer

3. **Labels are floats**: 0.0 (bad) or 1.0 (good)

4. **Prediction uses EOS token**: Last token score determines quality

5. **Must retrain if old model**: Old `num_labels=2` models won't work

---

**Next Steps After Retraining**:
1. Verify model passes `verify_trained_model.py`
2. Preprocess PPO data for both cases
3. Run PPO training experiments
4. Compare Case 1 vs Case 2 performance! 🚀

