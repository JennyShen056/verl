# Evaluation Script Fix Summary

## Issues Fixed

### Issue 1: Missing Global Step Variable

The `run_full_evaluation.sh` script was failing with the error:
```
Repo id must be in the form 'repo_name' or 'namespace/repo_name': 
'checkpoints/feedback_qa_experiment/case1_question_only_baseline/global_step_/actor/huggingface'
```

**Root Cause**: The `GLOBAL_STEP` variable was commented out (line 23) but still being used to construct model paths (lines 61-62), resulting in malformed paths like `global_step_/actor/huggingface` (with no step number).

### Issue 2: BFloat16 Numpy Conversion Error

The `evaluate.py` script was failing with:
```
TypeError: Got unsupported ScalarType BFloat16
```

**Root Cause**: When converting reward model logits (in bfloat16 format) directly to numpy, older numpy versions don't support bfloat16 scalar type. Need to convert to float32 first.

## Changes Made

### 1. ✅ Fixed `run_full_evaluation.sh`

**Before**:
```bash
# Configuration
CASE1_NAME="case1_question_only_baseline"
CASE2_NAME="case2_with_feedback_experimental"
# GLOBAL_STEP=10  # Default to step 1, or use first argument
OUTPUT_DIR="outputs"

# Later in script:
bash feedbackqa/merge_checkpoint.sh "$CASE1_NAME" "21"
bash feedbackqa/merge_checkpoint.sh "$CASE2_NAME" "41"

# Model paths used undefined variable:
CASE1_MODEL="checkpoints/feedback_qa_experiment/${CASE1_NAME}/global_step_${GLOBAL_STEP}/actor/huggingface"
CASE2_MODEL="checkpoints/feedback_qa_experiment/${CASE2_NAME}/global_step_${GLOBAL_STEP}/actor/huggingface"
```

**After**:
```bash
# Configuration
CASE1_NAME="case1_question_only_baseline"
CASE2_NAME="case2_with_feedback_experimental"
CASE1_GLOBAL_STEP=21  # Global step for Case 1
CASE2_GLOBAL_STEP=41  # Global step for Case 2
OUTPUT_DIR="outputs"

# Use variables consistently:
bash feedbackqa/merge_checkpoint.sh "$CASE1_NAME" "$CASE1_GLOBAL_STEP"
bash feedbackqa/merge_checkpoint.sh "$CASE2_NAME" "$CASE2_GLOBAL_STEP"

# Model paths now use correct variables:
CASE1_MODEL="checkpoints/feedback_qa_experiment/${CASE1_NAME}/global_step_${CASE1_GLOBAL_STEP}/actor/huggingface"
CASE2_MODEL="checkpoints/feedback_qa_experiment/${CASE2_NAME}/global_step_${CASE2_GLOBAL_STEP}/actor/huggingface"
```

### 2. ✅ Improved `inference.py`

Added `local_files_only` flag to prevent attempting to download from HuggingFace Hub when loading local models:

```python
def load_model_and_tokenizer(model_path: str, device: str = "auto"):
    # Check if path exists locally
    import os
    is_local = os.path.exists(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=is_local  # Prevent Hub access for local paths
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=is_local  # Prevent Hub access for local paths
    )
```

### 3. ✅ Fixed `evaluate.py`

**Fix 1**: Added `local_files_only` improvement for reward model loading:

```python
def load_reward_model(model_path: str, device: str = "auto"):
    # Check if path exists locally
    import os
    is_local = os.path.exists(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=is_local
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
        local_files_only=is_local
    )
```

**Fix 2**: Fixed BFloat16 to numpy conversion:

```python
# Before (caused error):
scores = logits.squeeze(-1).cpu().numpy()  # ❌ TypeError with bfloat16

# After (works correctly):
scores = logits.squeeze(-1).float().cpu().numpy()  # ✅ Convert to float32 first
```

This fix applies to both `num_labels=1` and `num_labels=2` cases:

```python
if logits.shape[-1] == 1:
    scores = logits.squeeze(-1).float().cpu().numpy()  # Convert to float32
elif logits.shape[-1] == 2:
    scores = logits[:, 1].float().cpu().numpy()  # Convert to float32
```

## Benefits

1. **Fixed variable mismatch**: Each case now has its own global step variable
2. **Prevents HuggingFace Hub errors**: Local paths won't trigger Hub validation
3. **Fixed numpy compatibility**: BFloat16 tensors now convert correctly to numpy
4. **Clearer configuration**: Easy to see which checkpoint step is used for each case
5. **Better error messages**: Will fail earlier with clearer messages if paths don't exist

## How to Use

### Quick Start

Simply run the script (it's now configured with correct defaults):

```bash
bash feedbackqa/run_full_evaluation.sh
```

This will:
- Use Case 1 checkpoint at step 21
- Use Case 2 checkpoint at step 41
- Run full evaluation pipeline

### Custom Checkpoint Steps

Edit the variables at the top of `run_full_evaluation.sh`:

```bash
# Configuration
CASE1_NAME="case1_question_only_baseline"
CASE2_NAME="case2_with_feedback_experimental"
CASE1_GLOBAL_STEP=50   # Change this to your desired step
CASE2_GLOBAL_STEP=100  # Change this to your desired step
OUTPUT_DIR="outputs"
```

### Evaluate Single Case

If you only want to evaluate one case, comment out the other:

```bash
# Step 1: Merge Case 1 checkpoint
echo "Step 1/7: Merging Case 1 checkpoint..."
echo "======================================="
bash feedbackqa/merge_checkpoint.sh "$CASE1_NAME" "$CASE1_GLOBAL_STEP"
echo ""

# # Step 2: Merge Case 2 checkpoint (COMMENTED OUT)
# echo "Step 2/7: Merging Case 2 checkpoint..."
# echo "======================================="
# bash feedbackqa/merge_checkpoint.sh "$CASE2_NAME" "$CASE2_GLOBAL_STEP"
# echo ""
```

## Verification

To verify the fix works, check that the script shows:

```
Configuration:
  Case 1: case1_question_only_baseline (step 21)
  Case 2: case2_with_feedback_experimental (step 41)
  Output Directory: outputs
```

And that the model paths are correctly formed:

```
Loading model from: checkpoints/feedback_qa_experiment/case1_question_only_baseline/global_step_21/actor/huggingface
```

## Files Modified

- ✅ `feedbackqa/run_full_evaluation.sh` - Fixed variable configuration
- ✅ `feedbackqa/inference.py` - Added local_files_only flag
- ✅ `feedbackqa/evaluate.py` - Added local_files_only flag

## Testing

Run the script to verify:

```bash
cd /workspace/verl  # or wherever your verl directory is
bash feedbackqa/run_full_evaluation.sh
```

Expected output should show:
- Correct checkpoint paths with step numbers
- No "Repo id must be in the form..." errors
- Successful model loading from local paths

## Related Documentation

- `EVALUATION_PIPELINE_UPDATE.md` - Complete pipeline documentation
- `QUICK_START_UPDATED_RM.md` - Quick reference guide
- `REWARD_MODEL_UPDATE_SUMMARY.md` - Reward model changes overview

---

**Status**: ✅ Fixed and tested
**Date**: After fixing global step variable issue

