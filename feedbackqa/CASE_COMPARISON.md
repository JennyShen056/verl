# Case 1 vs Case 2: Complete Comparison

## 🎯 Summary

| Aspect | Case 1: Question Only | Case 2: With Feedback |
|--------|----------------------|----------------------|
| **Policy Input** | Question only | Question + Previous Answer + Feedback |
| **RM Input** | Question + Generated Response | Question + Generated Response |
| **Data Field** | No `prompt_for_rm` | Has `prompt_for_rm` (question-only) |
| **Purpose** | Baseline | Experimental (feedback learning) |

---

## 📁 **Case 1: Question Only (Baseline)**

### Data Format
```python
{
    "prompt": [{"role": "user", "content": "What is X?"}],  # Question only
    "reward_model": {
        "style": "model",
        "ground_truth": "...",
        # No prompt_for_rm - not needed since prompt is already question-only
    }
}
```

### What Happens
1. **Policy sees**: Just the question
2. **Policy generates**: Response from scratch
3. **RM evaluates**: Question + Generated Response
4. **Code path**: Uses default `input_ids` (no special processing)

### Files
- Preprocessing: `preprocess_ppo_case1_question_only.py`
- Training: `run_ppo_case1_question_only.sh`
- Data: `~/data/feedback_qa_ppo/case1_question_only/`

---

## 📁 **Case 2: With Feedback (Experimental)**

### Data Format
```python
{
    "prompt": [
        {
            "role": "user", 
            "content": "Previous Answer: ...\nFeedback: ...\nNow answer: What is X?"
        }
    ],  # Full context
    "reward_model": {
        "style": "model",
        "ground_truth": "...",
        "prompt_for_rm": [{"role": "user", "content": "What is X?"}]  # Question only
    }
}
```

### What Happens
1. **Policy sees**: Question + Previous Answer + Feedback
2. **Policy generates**: Response based on full context
3. **RM evaluates**: Question + Generated Response (rebuilt using `prompt_for_rm`)
4. **Code path**: Uses `_build_rm_input_with_question_only()` method

### Files
- Preprocessing: `preprocess_ppo_case2_with_feedback.py`
- Training: `run_ppo_case2_with_feedback.sh`
- Data: `~/data/feedback_qa_ppo/case2_with_feedback/`

---

## 🔧 **Impact of Recent Changes**

### ✅ Both Cases Still Work!

#### **For Case 1** (No Changes Needed)
- ✅ No `prompt_for_rm` in data → uses default behavior
- ✅ RM uses full `input_ids` as-is
- ✅ Backward compatible

#### **For Case 2** (New Functionality)
- ✅ Has `prompt_for_rm` in data → activates question-only mode
- ✅ RM rebuilds input using question-only prompt
- ✅ Policy still learns from full context

### What Changed
1. **`fsdp_workers.py`**: 
   - Always loads RM tokenizer now (harmless for Case 1)
   - Detects `prompt_for_rm` automatically
   - Falls back to default if not present

2. **Both run scripts**: 
   - Removed `input_tokenizer=null` (not needed anymore)
   - Commented out TRAINED_RM_PATH check (using FsfairX model)

---

## 🚀 **How to Run Both Cases**

### Case 1 (Baseline)
```bash
cd /Users/jennyshen/verl

# Preprocess (if not done)
python feedbackqa/preprocess_ppo_case1_question_only.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case1_question_only

# Run training
bash feedbackqa/run_ppo_case1_question_only.sh
```

### Case 2 (Experimental)
```bash
cd /Users/jennyshen/verl

# Preprocess (required - has new prompt_for_rm field)
python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case2_with_feedback

# Run training
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

---

## 📊 **Expected Logs**

### Case 1 Logs
```
# No special RM debug messages
# Just regular training output
```

### Case 2 Logs
```
# You'll see this confirming question-only mode:
[RM Question-Only Mode] chat: <user>What is X?</user><assistant>...</assistant>
```

---

## 🔍 **Quick Verification**

### Check Case 1 Data
```python
import pandas as pd
df = pd.read_parquet("~/data/feedback_qa_ppo/case1_question_only/train.parquet")
print("Has prompt_for_rm:", "prompt_for_rm" in df.iloc[0].get("reward_model", {}))
# Should print: False
```

### Check Case 2 Data
```python
import pandas as pd
df = pd.read_parquet("~/data/feedback_qa_ppo/case2_with_feedback/train.parquet")
print("Has prompt_for_rm:", "prompt_for_rm" in df.iloc[0]["reward_model"])
# Should print: True
```

---

## 💡 **Key Takeaways**

1. ✅ **Case 1 is unaffected** - uses default behavior
2. ✅ **Case 2 uses new feature** - question-only RM input
3. ✅ **Both can run in parallel** - compare results in W&B
4. ✅ **Automatic detection** - no manual config needed
5. ✅ **Backward compatible** - old data still works

---

## 🎓 **Hypothesis Testing**

**Research Question**: Does seeing feedback examples help the policy learn better?

- **Case 1 (Control)**: Policy learns from RM scores alone
- **Case 2 (Treatment)**: Policy learns from RM scores + feedback examples

**Compare**:
- Final reward scores
- Sample quality
- Training speed
- Generalization to test set

Both use the **same RM** (FsfairX-LLaMA3-RM-v0.1) for fair comparison!

---

## 📚 Related Files

- 📖 **Setup Guide**: `QUESTION_ONLY_RM_GUIDE.md`
- 📋 **Changes**: `CHANGES_SUMMARY.md`
- 🔬 **This Document**: `CASE_COMPARISON.md`

