# Summary of Changes: Question-Only Reward Model Input

## Problem
You wanted the **policy model** to learn from full context (question + previous answer + feedback), but the **reward model** should only evaluate based on the question + generated answer (without previous context).

## Solution
Modified the codebase to support storing and using two different prompts:
1. **Full prompt** → Used by policy during rollout
2. **Question-only prompt** → Used by reward model during evaluation

---

## Files Modified

### 1. `feedbackqa/preprocess_ppo_case2_with_feedback.py`
**Lines 116-133**: Added `prompt_for_rm` field

**What it does**:
- Stores question-only prompt in `reward_model.prompt_for_rm`
- Full context remains in `prompt` for policy training

```python
"prompt_for_rm": [{"role": "user", "content": question}]  # Question only
```

### 2. `verl/workers/fsdp_workers.py`
**Lines 1868-1942**: Added `_build_rm_input_with_question_only()` method

**Lines 1955-1979**: Modified `compute_rm_score()` to detect and use question-only prompts

**What it does**:
- Automatically detects if `prompt_for_rm` exists
- Rebuilds input as: question + generated_response
- Falls back to regular prompt if `prompt_for_rm` not found

### 3. `verl/workers/reward_model/megatron/reward_model.py`
**Lines 131-220**: Added `build_question_only_input()` method

**Lines 227-240**: Modified `compute_reward()` to detect and use question-only prompts

**Line 290**: Updated to restore original values for both question-only and different tokenizer modes

**What it does**:
- Same as FSDP but for Megatron backend
- Handles Megatron-specific tokenizer differences

---

## How to Use

### Step 1: Reprocess Your Data
```bash
python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --test_file feedbackqa/feedback_test_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case2_with_feedback
```

### Step 2: Run Training (No Changes Needed!)
```bash
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

The system will automatically:
1. Detect `prompt_for_rm` in the data
2. Use full prompt for policy rollout
3. Use question-only prompt for reward model evaluation

---

## Verification

During training, look for these debug messages:

**FSDP Backend:**
```
[RM Question-Only Mode] chat: <user>What is X?</user><assistant>...</assistant>
```

**Megatron Backend:**
```
[Megatron RM Question-Only Mode] chat: ...
```

This confirms the reward model is using question-only inputs!

---

## Data Flow Diagram

```
Dataset
  ↓
  ├─→ prompt: [Full Context]           → Policy Model (Rollout)
  │                                           ↓
  │                                    [Generates Response]
  │                                           ↓
  └─→ reward_model.prompt_for_rm:     → Reward Model
      [Question Only] + [Response]         (Evaluation)
```

---

## Benefits

✅ Policy learns from rich feedback context  
✅ Reward model provides unbiased evaluation  
✅ Automatic detection (no config changes)  
✅ Backward compatible (falls back if field missing)  
✅ Works with both FSDP and Megatron backends  

---

## Testing

Test on a small dataset first:
```bash
# Create small test set
head -n 100 feedbackqa/feedback_train_ppo.json > feedbackqa/feedback_train_small.json

# Preprocess
python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_small.json \
    --local_save_dir ~/data/feedback_qa_test

# Check output
python -c "
import pandas as pd
df = pd.read_parquet('~/data/feedback_qa_test/train.parquet')
print('Full prompt:', df.iloc[0]['prompt'])
print('RM prompt:', df.iloc[0]['reward_model']['prompt_for_rm'])
"
```

---

## Troubleshooting

### Problem: RM still sees full context
**Check**: Did you reprocess the data with the updated script?
**Check**: Does your parquet file have `prompt_for_rm` field?

### Problem: Training crashes
**Check**: Are you using the reprocessed data files?
**Check**: Check logs for error messages

### Problem: No debug messages
**Check**: Make sure you're looking at the correct process logs (rank 0)

---

## Related Files

📖 **Full Guide**: `feedbackqa/QUESTION_ONLY_RM_GUIDE.md`  
🔧 **Preprocessing**: `feedbackqa/preprocess_ppo_case2_with_feedback.py`  
⚙️ **FSDP Worker**: `verl/workers/fsdp_workers.py`  
⚙️ **Megatron RM**: `verl/workers/reward_model/megatron/reward_model.py`  

---

## Questions?

See `QUESTION_ONLY_RM_GUIDE.md` for detailed documentation!
