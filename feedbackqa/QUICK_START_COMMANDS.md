# Quick Start Commands

## ✅ Ready to Run!

Your trained reward model is at: `/workspace/verl/feedbackqa/feedback_qa_reward_model`

Both Case 1 and Case 2 are now configured to use it.

---

## 🚀 Run Case 1 (Question Only - Baseline)

```bash
cd /workspace/verl

# Preprocess data if not already done
python feedbackqa/preprocess_ppo_case1_question_only.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case1_question_only

# Run training
bash feedbackqa/run_ppo_case1_question_only.sh
```

---

## 🚀 Run Case 2 (With Feedback - Experimental)

```bash
cd /workspace/verl

# Preprocess data (REQUIRED - includes prompt_for_rm)
python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case2_with_feedback

# Run training
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

---

## 📋 What's Configured

### Both Cases Use:
- **Reward Model**: `/workspace/verl/feedbackqa/feedback_qa_reward_model` (your trained model)
- **Policy Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Hardware**: 8 GPUs

### Case 1 Settings:
- **Input**: Question only
- **Epochs**: 3
- **Batch Size**: 256
- **Experiment**: `case1_question_only_baseline`

### Case 2 Settings:
- **Input**: Question + Previous Answer + Feedback
- **RM Input**: Question only (via `prompt_for_rm`)
- **Epochs**: 1
- **Batch Size**: 128
- **Experiment**: `case2_with_feedback_experimental`

---

## 🔍 Verify Data is Ready

### Check Case 1 Data
```bash
ls -lh ~/data/feedback_qa_ppo/case1_question_only/
# Should see: train.parquet, valid.parquet
```

### Check Case 2 Data
```bash
ls -lh ~/data/feedback_qa_ppo/case2_with_feedback/
# Should see: train.parquet, valid.parquet
```

### Check Reward Model
```bash
ls -lh /workspace/verl/feedbackqa/feedback_qa_reward_model/
# Should see: pytorch_model.bin, config.json, etc.
```

---

## 📊 Monitor Training

### Watch Logs
```bash
# Case 1 - look for regular training progress
tail -f <log_file>

# Case 2 - look for this line confirming question-only RM mode:
tail -f <log_file> | grep "RM Question-Only Mode"
```

### W&B Dashboard
- **Project**: `feedback_qa_experiment`
- **Experiments**: 
  - `case1_question_only_baseline`
  - `case2_with_feedback_experimental`

---

## ⚠️ Troubleshooting

### If Case 1 fails to start:
```bash
# Check if data exists
ls ~/data/feedback_qa_ppo/case1_question_only/train.parquet

# If not, preprocess first
python feedbackqa/preprocess_ppo_case1_question_only.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case1_question_only
```

### If Case 2 fails to start:
```bash
# Check if data with prompt_for_rm exists
python -c "
import pandas as pd
df = pd.read_parquet('$HOME/data/feedback_qa_ppo/case2_with_feedback/train.parquet')
print('Has prompt_for_rm:', 'prompt_for_rm' in df.iloc[0]['reward_model'])
"

# If False, reprocess with updated script
python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case2_with_feedback
```

### If reward model not found:
```bash
# Verify it exists
ls /workspace/verl/feedbackqa/feedback_qa_reward_model/

# If not, train it first
python feedbackqa/rm_train.py
```

---

## 🎯 Expected Results

### Case 1 (Baseline)
- Policy learns from RM feedback alone
- No examples shown
- Clean baseline to compare against

### Case 2 (Experimental)
- Policy sees feedback examples
- Should show: `[RM Question-Only Mode] chat: ...` in logs
- Tests if examples improve learning

### Comparison
Both use the **same RM** for fair comparison!
Compare final rewards, sample quality, and convergence speed in W&B.

---

## 📚 More Info

- **Setup Guide**: `QUESTION_ONLY_RM_GUIDE.md`
- **Comparison**: `CASE_COMPARISON.md`
- **Changes**: `CHANGES_SUMMARY.md`

