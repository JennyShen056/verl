# Question-Only Reward Model Input Guide

## Overview

This guide explains how to use different prompts for **policy training** vs **reward model evaluation** in your PPO training setup.

### Use Case

You want to:
- **Policy Model**: Learn from full context (question + previous answer + feedback)
- **Reward Model**: Evaluate based only on (question + generated answer)

This separation allows the policy to learn from examples while the reward model provides unbiased evaluation.

---

## What Was Changed

### 1. Preprocessing Script (`preprocess_ppo_case2_with_feedback.py`)

**Added**: A `prompt_for_rm` field in the reward_model metadata that contains only the question.

```python
"reward_model": {
    "style": "model",
    "ground_truth": section_content,
    "feedback": feedback,
    "rating": rating,
    "prompt_for_rm": question_only_prompt,  # ← NEW: Question-only for RM
},
```

### 2. FSDP Reward Model Worker (`verl/workers/fsdp_workers.py`)

**Added**: 
- `_build_rm_input_with_question_only()` method that rebuilds the input using only the question
- Logic in `compute_rm_score()` to detect and use question-only prompts

### 3. Megatron Reward Model (`verl/workers/reward_model/megatron/reward_model.py`)

**Added**:
- `build_question_only_input()` method for Megatron backend
- Logic in `compute_reward()` to detect and use question-only prompts

---

## How It Works

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Dataset provides TWO versions:                           │
│    - Full prompt (for policy): Question + Prev Answer + FB  │
│    - Question-only (for RM): Just the question             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. Policy Model Training (Rollout)                          │
│    Input: Full prompt with context                          │
│    Generates: Response based on full context                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. Reward Model Evaluation                                  │
│    Input: Question-only + Generated Response                │
│    Output: Score without seeing previous answers            │
└─────────────────────────────────────────────────────────────┘
```

### Automatic Detection

The reward model worker automatically detects if `prompt_for_rm` exists:

```python
if "reward_model" in data.non_tensor_batch:
    if "prompt_for_rm" in data.non_tensor_batch["reward_model"][0]:
        # Use question-only mode automatically
        use_question_only = True
```

---

## How to Use

### Step 1: Reprocess Your Data

Run the preprocessing script to generate data with question-only prompts:

```bash
cd /Users/jennyshen/verl

python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --test_file feedbackqa/feedback_test_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case2_with_feedback \
    --seed 42
```

### Step 2: Run PPO Training

No changes needed to your training script! The system automatically detects `prompt_for_rm` and uses it.

```bash
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

### Step 3: Verify It's Working

During training, you should see debug output like:

```
[RM Question-Only Mode] chat: <user>
What is the capital of France?
</user>
<assistant>
The capital of France is Paris.
</assistant>
```

This confirms the reward model is only seeing the question (not previous answers/feedback).

---

## Data Format

### Full Dataset Entry

```python
{
    "data_source": "feedback_qa",
    "prompt": [
        {
            "role": "user",
            "content": "Here is a previous answer with feedback:\n\n" +
                      "Question: What is X?\n\n" +
                      "Previous Answer: ...\n\n" +
                      "Feedback: ... (Rating: 3)\n\n" +
                      "Now, answer the same question:\n\n" +
                      "Question: What is X?"
        }
    ],
    "reward_model": {
        "style": "model",
        "ground_truth": "...",
        "feedback": "...",
        "rating": "3",
        "prompt_for_rm": [                    # ← Question-only version
            {
                "role": "user",
                "content": "What is X?"
            }
        ]
    }
}
```

### What Policy Sees

```
Input Prompt:
  "Here is a previous answer with feedback:
   
   Question: What is X?
   Previous Answer: Y
   Feedback: Good but could be better (Rating: 3)
   
   Now answer the same question:
   Question: What is X?"

Generated Response:
  "X is Z because..."
```

### What Reward Model Sees

```
Input: "What is X?"
Generated Response: "X is Z because..."

Combined for RM: "<user>What is X?</user><assistant>X is Z because...</assistant>"
```

---

## Benefits

1. **Unbiased Reward Signals**: RM evaluates responses without being influenced by previous examples
2. **Rich Policy Learning**: Policy still learns from feedback context
3. **Flexible**: Falls back gracefully if `prompt_for_rm` is not present
4. **Automatic**: No manual intervention needed during training

---

## Debugging

### Check if Question-Only Mode is Active

Look for this line in your training logs:

```
[RM Question-Only Mode] chat: ...
```

or

```
[Megatron RM Question-Only Mode] chat: ...
```

### Verify Data Format

Check your preprocessed parquet file:

```python
import pandas as pd

df = pd.read_parquet("~/data/feedback_qa_ppo/case2_with_feedback/train.parquet")

# Check first example
example = df.iloc[0]
print("Full prompt:", example['prompt'])
print("RM prompt:", example['reward_model']['prompt_for_rm'])
```

### Common Issues

**Issue**: Reward model still sees full context

**Solution**: Make sure:
1. You reprocessed the data with the updated script
2. The `prompt_for_rm` field exists in your dataset
3. Your training script points to the newly processed data

---

## Backward Compatibility

If `prompt_for_rm` is **not** present in the data:
- System automatically falls back to using the full `raw_prompt`
- No errors or crashes
- Works with existing datasets

---

## Technical Details

### FSDP Implementation

```python
def _build_rm_input_with_question_only(self, data: DataProto):
    # 1. Extract question-only prompt from metadata
    question_only_chat = data.non_tensor_batch["reward_model"][i]["prompt_for_rm"]
    
    # 2. Extract generated response
    response = decode(data.batch["responses"][i])
    
    # 3. Combine: question + response
    question_only_chat.append({"role": "assistant", "content": response})
    
    # 4. Tokenize and return
    return tokenized_input
```

### Megatron Implementation

Similar to FSDP but with additional handling for Megatron's tokenizer differences and parameter offloading.

---

## Example Output

### Training Log

```
Processing train split
============================================================
Loaded 5000 examples
Processing 5000 examples for Case 2 (With Feedback)...

Dataset Statistics:
  Total examples: 5000
  Split: train
  Case: With Previous Answer to Same Question (Feedback-Augmented)
  Examples with previous answer: 4873 (97.5%)
  Examples without previous answer: 127 (2.5%)

Example prompt (with previous answer to same question):
  Here is a previous answer to this question with feedback:

  Question: What is the main theme of Section 3?

  Previous Answer: The section discusses various topics...

  Feedback: Good analysis but missing key details (Rating: 3)

  Now, please answer the same question:

  Question: What is the main theme of Section 3?
```

### During PPO Training

```
[RM Question-Only Mode] chat: <user>What is the main theme of Section 3?</user><assistant>The main theme is...</assistant>
```

This confirms the RM only sees the question, not the previous answer/feedback!

---

## Summary

✅ **Policy learns from**: Question + Previous Answer + Feedback  
✅ **Reward model evaluates**: Question + Generated Answer only  
✅ **Automatic detection**: No config changes needed  
✅ **Backward compatible**: Works with old datasets too  

You're all set! Just reprocess your data and run training as usual.

