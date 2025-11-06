# Quick Start: Updated Reward Model Training & Evaluation

## What Changed?

Your reward model training has been updated from **binary classification (num_labels=2)** to **scalar regression (num_labels=1)** to be compatible with verl's PPO training.

## 🚀 Quick Start (3 Steps)

### Step 1: Train New Reward Model

```bash
cd feedbackqa

python rm_train.py \
    --train_file feedback_train_rm.json \
    --valid_file feedback_valid_rm.json \
    --test_file feedback_test_rm.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./feedback_qa_rm_v2 \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 2e-5
```

⏱️ **Time**: ~2-4 hours depending on hardware

### Step 2: Verify Compatibility

```bash
python verify_trained_model.py --model_path ./feedback_qa_rm_v2/final_model
```

✅ **Expected**: "SUCCESS! Your model is verl-compatible!"

### Step 3: Use with verl PPO

Update your PPO training script:

```bash
# In run_ppo_case1_question_only.sh or run_ppo_case2_with_feedback.sh
reward_model.enable=True \
reward_model.model.path=./feedbackqa/feedback_qa_rm_v2/final_model \
```

## 📊 Evaluation Pipeline

### Run Full Evaluation

```bash
bash run_full_evaluation.sh
```

This automatically:
1. ✅ Detects model format (num_labels=1 or 2)
2. ✅ Merges PPO checkpoints
3. ✅ Generates predictions
4. ✅ Computes reward scores
5. ✅ Produces comparison statistics

### Manual Evaluation

```bash
# Evaluate specific predictions
python evaluate.py \
    --predictions_file outputs/case1_predictions.json \
    --reward_model_path ./feedback_qa_rm_v2/final_model \
    --output_file outputs/case1_evaluation.json \
    --batch_size 8
```

## 🔍 What to Check

### Model Format Indicators

**Old format (num_labels=2) - NOT verl-compatible**:
```
num_labels: 2
Format: Binary classification (NOT verl-compatible)
Output shape: [batch_size, 2]
```

**New format (num_labels=1) - ✅ verl-compatible**:
```
num_labels: 1
Format: Scalar regression (verl-compatible)
Output shape: [batch_size, 1]
```

### Verification Checklist

Before using with verl:
- [ ] `verify_trained_model.py` passes all checks
- [ ] Output shows "num_labels: 1"
- [ ] Output shows "problem_type: regression"
- [ ] Output shape is `[batch_size, 1]`
- [ ] Format shows "Scalar regression (verl-compatible)"

## 📈 Expected Reward Ranges

### Training Targets
- **Not relevant** (label 0) → **-1.0**
- **Relevant** (label 1) → **+1.0**

### Actual Predictions
- **Theoretical range**: (-∞, +∞) unbounded
- **Practical range**: typically [-2.0, +2.0]
- **Decision threshold**: 0.0
  - Score > 0 → classified as relevant
  - Score < 0 → classified as not relevant

### Why Unbounded is Good
- Allows model to express varying confidence levels
- Better for RL - distinguishes strong vs weak preferences
- verl normalizes internally anyway:
  ```python
  advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
  ```

## 🔄 Key File Changes

| File | What Changed |
|------|-------------|
| `rm_train.py` | num_labels: 2→1, classification→regression |
| `evaluate.py` | Auto-detects format, supports both |
| `verify_trained_model.py` | Fixed architecture check, added tests |

## 💡 Understanding the Formats

### Your Old Approach (num_labels=2)
```
Input → Model → [logit_0, logit_1]
                      ↓
                  Softmax
                      ↓
                [prob_0, prob_1]
                      ↓
              Use prob_1 as score
```
❌ **Problem**: verl expects 1 value, not 2

### Your New Approach (num_labels=1)
```
Input → Model → [reward_score]
                      ↓
              Direct scalar output
```
✅ **Perfect**: Exactly what verl needs!

### RLHFlow Bradley-Terry (reference)
```
Input_chosen → Model → [reward_chosen]
Input_rejected → Model → [reward_rejected]
                      ↓
Loss = -log(sigmoid(reward_chosen - reward_rejected))
```
✅ **Also num_labels=1**: Different training method, same output format

## 🎯 Common Issues

### Issue: "Expected num_labels=1, got 2"
**Solution**: Retrain with updated `rm_train.py`

### Issue: verl PPO fails with shape error
**Solution**: Run `verify_trained_model.py` first

### Issue: Evaluation works but verl fails
**Reason**: Evaluation supports both formats, verl only supports num_labels=1
**Solution**: Always verify before using with verl

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| `REWARD_MODEL_FORMAT_EXPLANATION.md` | Detailed technical explanation |
| `EVALUATION_PIPELINE_UPDATE.md` | Full pipeline update guide |
| `QUICK_START_UPDATED_RM.md` | This file - quick reference |

## 🔬 Testing Your Model

### Quick Test
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "./feedback_qa_rm_v2/final_model"
)

print(f"num_labels: {model.config.num_labels}")
print(f"out_features: {model.classifier.out_features}")

# Should output:
# num_labels: 1
# out_features: 1
```

### Full Verification
```bash
python verify_trained_model.py --model_path ./feedback_qa_rm_v2/final_model
```

## ⚡ Next Steps

1. **Train** new reward model with `rm_train.py` ✅
2. **Verify** with `verify_trained_model.py` ✅
3. **Update** PPO scripts to use new model path ✅
4. **Run** PPO training with verified reward model ✅
5. **Evaluate** results with `run_full_evaluation.sh` ✅

## 🎉 Benefits

- ✅ **verl-compatible**: Works with PPO training
- ✅ **Better for RL**: Continuous reward scores with confidence
- ✅ **Same data**: No need to create preference pairs
- ✅ **Backward compatible**: Evaluation still works with old models
- ✅ **Easy migration**: Just retrain and verify

## Questions?

- Check `REWARD_MODEL_FORMAT_EXPLANATION.md` for technical details
- Check `EVALUATION_PIPELINE_UPDATE.md` for pipeline specifics
- Run `verify_trained_model.py --help` for verification options
- Run `python rm_train.py --help` for training options

---

**Last Updated**: After reward model format update to num_labels=1

