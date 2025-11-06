# Training Scripts Updated Based on Analysis

**Date**: Based on prompt/response length analysis of `feedback_train_ppo.json`  
**Analysis Source**: `analyze_prompt_response_lengths.py` results

---

## Summary of Changes

Both training scripts have been updated with **data-driven parameters** from the actual token length analysis.

---

## Case 1: Question Only (`run_ppo_case1_question_only.sh`)

### Updated Parameters

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `data.max_prompt_length` | 1024 | **512** | Prompts max out at 37 tokens (mean 15.4) |
| `data.max_response_length` | 512 | **565** | 95th percentile is 514 tokens, +10% buffer |
| `data.train_batch_size` | 256 | **256** | ✅ No change - short prompts allow large batches |

### Key Stats from Analysis
```
Question prompts:
  Mean:           15.4 tokens
  Median:         15 tokens  
  95th percentile: 24 tokens
  Max:            37 tokens

Total sequence (prompt + response):
  Mean:           211.7 tokens
  95th percentile: 531 tokens
```

### Why These Changes?

- **`max_prompt_length=512`**: Way more than needed (max is 37), but provides comfortable buffer
- **`max_response_length=565`**: Covers 95% of responses (514) + 10% buffer
- **`train_batch_size=256`**: Can maintain large batches due to short prompts

---

## Case 2: With Previous Answer + Feedback (`run_ppo_case2_with_feedback.sh`)

### Updated Parameters

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| `data.max_prompt_length` | 2048 | **1024** | 95th percentile is 635 tokens |
| `data.max_response_length` | 512 | **565** | Same as Case 1 (responses identical) |
| `data.train_batch_size` | 256 | **128** | 50% reduction due to 2x longer sequences |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 128 | **64** | Proportional reduction |

### Key Stats from Analysis
```
Feedback-augmented prompts:
  Mean:           288.6 tokens
  Median:         232 tokens
  95th percentile: 635 tokens
  Max:            1,613 tokens

Total sequence (prompt + response):
  Mean:           484.9 tokens
  95th percentile: 1,111 tokens
  Max:            3,137 tokens
```

### Why These Changes?

- **`max_prompt_length=1024`**: Covers 95% of prompts (635) with ~60% headroom
  - If you hit truncation warnings, increase to 1536 or 2048
  - Current setting truncates only ~5% of longest examples
  
- **`max_response_length=565`**: Same as Case 1 (answers are identical between cases)

- **`train_batch_size=128`**: **CRITICAL CHANGE**
  - Case 2 has 2.3x more tokens per example
  - Reduced from 256 to 128 (50% reduction) to fit in memory
  - This matches the memory increase from longer prompts
  
- **`ppo_mini_batch_size=64`**: Proportional reduction (was 128, now 64)

---

## Comparison: Case 1 vs Case 2

### Memory per Example

| Metric | Case 1 | Case 2 | Ratio |
|--------|--------|--------|-------|
| Mean prompt | 15 tokens | 289 tokens | **19.3x** |
| Mean response | 196 tokens | 196 tokens | 1.0x |
| **Mean total** | **212 tokens** | **485 tokens** | **2.3x** |

### Training Throughput

| Metric | Case 1 | Case 2 | Impact |
|--------|--------|--------|--------|
| Batch size | 256 | 128 | 0.5x |
| Tokens/example | 212 | 485 | 2.3x |
| **Relative speed** | **1.0x** | **~0.43x** | **2.3x slower** |

**Expected**: Case 2 will take ~2.3x longer to train than Case 1 for the same number of examples.

---

## Configuration Summary

### Case 1: Question Only
```bash
# Optimized for short prompts (mean 15 tokens)
data.max_prompt_length=512        # Generous buffer
data.max_response_length=565      # Covers 95th percentile + buffer
data.train_batch_size=256         # Large batches OK (short prompts)
```

### Case 2: With Feedback
```bash
# Optimized for feedback-augmented prompts (mean 289 tokens)
data.max_prompt_length=1024       # Covers 95th percentile (635)
data.max_response_length=565      # Same as Case 1
data.train_batch_size=128         # Reduced 50% for memory
ppo_mini_batch_size=64           # Also reduced proportionally
```

---

## Training Time Estimates

Assuming **8 x A100 (80GB) GPUs**, **5,400 training examples**, **10 epochs**:

### Case 1: Question Only
- Sequences per step: 256 × 8 = **2,048**
- Steps per epoch: 5,400 / 2,048 = **~2.6 steps**
- Total steps (10 epochs): **~26 steps**
- Estimated time: **~30-60 minutes** (depending on hardware)

### Case 2: With Feedback
- Sequences per step: 128 × 8 = **1,024**
- Steps per epoch: 5,400 / 1,024 = **~5.3 steps**
- Total steps (10 epochs): **~53 steps**
- Estimated time: **~90-150 minutes** (2-3x longer due to longer sequences)

**Note**: These are rough estimates. Actual time depends on:
- GPU utilization
- Network latency (if multi-node)
- Reward model inference speed
- Dataset complexity

---

## What If You Hit OOM (Out of Memory)?

### For Case 2 (Most Likely)

If you get OOM errors, try these in order:

1. **Reduce batch size**
   ```bash
   data.train_batch_size=64  # or even 32
   actor_rollout_ref.actor.ppo_mini_batch_size=32
   ```

2. **Reduce max_prompt_length** (truncate longer examples)
   ```bash
   data.max_prompt_length=800  # Covers ~90% instead of 95%
   ```

3. **Enable CPU offloading** (slower but uses less GPU memory)
   ```bash
   actor_rollout_ref.actor.fsdp_config.param_offload=True
   actor_rollout_ref.actor.fsdp_config.cpu_offload=True
   ```

4. **Reduce micro batch sizes**
   ```bash
   actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2  # down from 4
   critic.ppo_micro_batch_size_per_gpu=4  # down from 8
   ```

5. **Use smaller model for testing**
   ```bash
   POLICY_MODEL="meta-llama/Llama-3.2-1B-Instruct"  # Instead of 3B
   ```

---

## Validation: How to Check These Settings

After preprocessing, you can verify the actual data distributions:

```bash
# Run the inspection script
python feedbackqa/inspect_preprocessed_data.py \
    --case1_file ~/data/feedback_qa_ppo/case1_question_only/train.parquet \
    --case2_file ~/data/feedback_qa_ppo/case2_with_feedback/train.parquet \
    --tokenizer meta-llama/Llama-3.1-8B-Instruct \
    --num_samples 100
```

This will show you:
- Actual prompt length distributions in your data
- Example prompts from both cases
- Confirmation that parameters are appropriate

---

## Next Steps

1. ✅ **Scripts updated** with data-driven parameters
2. ⏭️ **Preprocess data** for both cases:
   ```bash
   python feedbackqa/preprocess_ppo_case1_question_only.py
   python feedbackqa/preprocess_ppo_case2_with_feedback.py
   ```

3. ⏭️ **Train Case 1** (baseline):
   ```bash
   bash feedbackqa/run_ppo_case1_question_only.sh
   ```

4. ⏭️ **Train Case 2** (experimental):
   ```bash
   bash feedbackqa/run_ppo_case2_with_feedback.sh
   ```

5. ⏭️ **Compare results** in Weights & Biases:
   - Project: `feedback_qa_experiment`
   - Runs: `case1_question_only_baseline` vs `case2_with_feedback_experimental`

---

## Monitoring During Training

### Key Metrics to Watch

1. **Truncation warnings**: Should be minimal (<5%)
   - If many warnings → increase `max_prompt_length`

2. **GPU memory usage**: Should be <90% per GPU
   - If consistently >95% → reduce batch size

3. **Training speed**:
   - Case 1: ~2-3 steps/minute expected
   - Case 2: ~1-1.5 steps/minute expected (slower)

4. **Reward scores**: 
   - Should increase over epochs
   - Compare Case 1 vs Case 2 to see if feedback helps!

---

## Key Insights

### ✅ Good News
- **100% context coverage**: All questions have multiple answers, so Case 2 works perfectly!
- **Reasonable lengths**: Even Case 2 fits comfortably in 1024 token window
- **Fair comparison**: Same response lengths, only prompt differs

### ⚠️ Important Notes
- **Case 2 is slower**: ~2.3x longer training time is expected
- **Batch size matters**: Don't use same batch size for both cases!
- **Memory is key**: Monitor GPU memory during training

### 🎯 Experiment Validity
- ✅ Same reward model for both cases
- ✅ Same response lengths (same answers)
- ✅ Only difference: feedback context during training
- ✅ Can directly compare Case 1 vs Case 2 performance

---

## Questions?

- Analysis details: `ANALYSIS_RESULTS.md`
- How to run analysis: `LENGTH_ANALYSIS_README.md`
- Experiment design: `EXPERIMENTAL_DESIGN.md`
- Step-by-step guide: `RUN_EXPERIMENT.md`

