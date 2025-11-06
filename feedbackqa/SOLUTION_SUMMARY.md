# Solution Summary: Reward Model Issue

## 🎯 Your Question

> "I trained the reward model but got an error about num_labels=2 instead of 1. How can I deal with this issue, and let me know how data is formatted during training?"

---

## ✅ Solution

### Issue Identified

Your reward model was trained with the OLD configuration:
- `AutoModelForSequenceClassification` 
- `num_labels=2` (binary classification)
- ❌ **Not compatible with verl PPO training**

### What I Fixed

Updated `rm_train.py` to use verl-compatible configuration:
- `AutoModelForTokenClassification` ✅
- `num_labels=1` (single scalar reward) ✅
- Float labels instead of int labels ✅
- EOS token scoring for rewards ✅

### What You Need to Do

**Delete old model and retrain:**

```bash
# 1. Remove old model
rm -rf ./feedback_qa_reward_model

# 2. Retrain with updated script
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model \
    --model_name meta-llama/Llama-3.1-8B-Instruct

# 3. Verify it works
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```

**Expected time**: 10-20 minutes on a single A100 GPU

---

## 📊 Data Format During Training

### Input Format (`feedback_train_rm.json`)

```json
[
  {
    "question": "How do I get help finding a job?",
    "section_content": "In this rapidly changing jobs market, it can be difficult...",
    "label": 1
  },
  {
    "question": "What are common interview mistakes?",
    "section_content": "Some mistakes include being late...",
    "label": 0
  }
]
```

**Fields**:
- `question`: User's question
- `section_content`: The answer/response
- `label`: `1` = good answer, `0` = bad answer

### Data Transformation Pipeline

```
1. Raw JSON
   ↓
2. Format with Chat Template
   <|begin_of_text|><|start_header_id|>user<|end_header_id|>
   How do I get help finding a job?<|eot_id|>
   <|start_header_id|>assistant<|end_header_id|>
   In this rapidly changing jobs market...<|eot_id|>
   ↓
3. Tokenize
   {
     "input_ids": [128000, 128006, 882, ..., 128009],
     "attention_mask": [1, 1, 1, ..., 1],
     "labels": 1.0  // Float, not int!
   }
   ↓
4. Model Forward Pass
   Input:  [batch_size, seq_len]
   Output: [batch_size, seq_len, 1]  // One score per token
   ↓
5. Extract EOS Token Score
   prediction = output[:, -1, 0]  // Last token score
   ↓
6. Compute Loss
   loss = MSE(prediction, label)
   ↓
7. Evaluation
   prob = sigmoid(prediction)
   predicted_class = 1 if prob > 0.5 else 0
   metrics: accuracy, precision, recall, F1, AUC
```

### Key Points

1. **Chat template format**: Structures Q&A as conversation
2. **Token classification**: Outputs one score per token
3. **EOS token prediction**: The last token's score is the reward
4. **Float labels**: Labels are 0.0 or 1.0 (not 0 or 1)
5. **MSE loss**: Regression loss (not cross-entropy)

---

## 🔄 Architecture Comparison

| Aspect | OLD (Not verl-compatible) | NEW (verl-compatible) ✅ |
|--------|---------------------------|--------------------------|
| Model Type | `AutoModelForSequenceClassification` | `AutoModelForTokenClassification` |
| `num_labels` | 2 | 1 |
| Output Shape | `[batch, 2]` | `[batch, seq_len, 1]` |
| Label Type | `int` (0 or 1) | `float` (0.0 or 1.0) |
| Loss | Cross-entropy | MSE |
| Prediction | `argmax(logits)` | `sigmoid(last_token)` |
| verl Compatible | ❌ NO | ✅ YES |

---

## 📚 Documentation Created

I created several detailed guides for you:

1. **`FIX_REWARD_MODEL_ISSUE.md`** ⭐
   - Quick 3-step fix guide
   - What changed in the script
   - Expected training metrics

2. **`RM_DATA_FORMAT_AND_RETRAINING.md`** 📖
   - Complete data format explanation
   - Detailed transformation pipeline
   - Troubleshooting guide

3. **`DATA_FLOW_DIAGRAM.md`** 🎨
   - Visual flow diagrams
   - Step-by-step transformations
   - Old vs new architecture comparison

---

## 🎓 Understanding the Data Flow

### Simple Version

```
Question + Answer → Format as conversation → Tokenize → Model scores each token
→ Take EOS token score → Compare to label (0 or 1) → Train to predict quality
```

### Why EOS Token?

The model learns to put the **overall quality score** in the **EOS (end-of-sequence) token** because:
- EOS token sees the complete Q&A pair
- Has full context to judge quality
- Natural summary point in the sequence
- verl extracts this as the reward during PPO

---

## ✅ After Retraining Checklist

Once your model is retrained:

- [ ] Verification passes (`num_labels=1` ✅)
- [ ] Training accuracy > 0.85
- [ ] Validation F1 > 0.85
- [ ] Model saved to `./feedback_qa_reward_model/final_model/`
- [ ] Ready for PPO training!

---

## 🚀 Next Steps

After you successfully retrain and verify the reward model:

1. **Preprocess PPO data**:
   ```bash
   python feedbackqa/preprocess_ppo_case1_question_only.py
   python feedbackqa/preprocess_ppo_case2_with_feedback.py
   ```

2. **Run PPO training**:
   ```bash
   bash feedbackqa/run_ppo_case1_question_only.sh  # Baseline
   bash feedbackqa/run_ppo_case2_with_feedback.sh  # Experimental
   ```

3. **Compare results** in Weights & Biases to see if feedback helps!

---

## 🔧 Quick Troubleshooting

### Verification still fails after retraining?

Check:
```bash
# Look at the config
cat ./feedback_qa_reward_model/final_model/config.json | grep num_labels
```

Should show: `"num_labels": 1`

If it still shows `"num_labels": 2`:
1. Make sure you deleted the old model folder
2. Check you're running the updated `rm_train.py`
3. Look for any errors during training

### Training OOM (out of memory)?

Reduce batch size:
```bash
python feedbackqa/rm_train.py ... --batch_size 2
```

### Low accuracy (<0.7)?

Train longer or adjust learning rate:
```bash
python feedbackqa/rm_train.py ... --epochs 5 --learning_rate 1e-5
```

---

## 📞 Quick Reference

### Files Updated

- ✏️ `rm_train.py` - Now verl-compatible
- ✨ `FIX_REWARD_MODEL_ISSUE.md` - Quick fix guide
- ✨ `RM_DATA_FORMAT_AND_RETRAINING.md` - Detailed explanation
- ✨ `DATA_FLOW_DIAGRAM.md` - Visual diagrams
- ✨ `SOLUTION_SUMMARY.md` - This file!

### Key Commands

```bash
# Retrain model
python feedbackqa/rm_train.py --train_file ... --model_name ...

# Verify model
python feedbackqa/verify_trained_model.py --model_path ...

# Preprocess PPO data
python feedbackqa/preprocess_ppo_case1_question_only.py
python feedbackqa/preprocess_ppo_case2_with_feedback.py

# Run PPO training
bash feedbackqa/run_ppo_case1_question_only.sh
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

---

## 💡 Key Takeaway

**The core issue**: Your model used the wrong architecture for verl.

**The fix**: Retrain with `AutoModelForTokenClassification` and `num_labels=1`.

**Why**: verl expects to extract a single reward score from the EOS token of a token classification model.

**Data format**: Q&A pairs → chat template → tokens → model scores each token → EOS token = reward

---

**Status**: ✅ Script updated, ready for you to retrain!

**Time to fix**: ~15 minutes retraining + verification

**Then**: Continue with your experiment to see if feedback helps models learn! 🎯

