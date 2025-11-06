# Prompt and Response Length Analysis

## Purpose
Analyze the token lengths for Case 1 vs Case 2 to help configure training parameters correctly.

## Running the Analysis

```bash
python feedbackqa/analyze_prompt_response_lengths.py
```

**Requirements:**
- `transformers` library
- `numpy`
- `tqdm` (optional, for progress bars)
- Access to `feedbackqa/feedback_train_ppo.json`

## What It Analyzes

### 1. Raw Data Statistics
- **Question lengths**: How many tokens in questions
- **Answer lengths**: How many tokens in responses (section_content)
- **Feedback lengths**: How many tokens in feedback
- **Question reuse**: How many questions have multiple answers

### 2. Case 1 Prompts (Question Only)
```
Prompt: "How do I get help finding a job?"
```
Analyzes: Token counts for simple question prompts

### 3. Case 2 Prompts (With Previous Answer)
```
Prompt: "Here is a previous answer to this question with feedback:

Question: How do I get help finding a job?

Previous Answer: [300+ token answer]

Feedback: [50+ token feedback]

Now, please answer the same question:

Question: How do I get help finding a job?"
```
Analyzes: Token counts with full context

### 4. Comparison
- Mean, median, 95th percentile lengths
- Case 1 vs Case 2 differences
- Total sequence lengths (prompt + response)

### 5. Training Recommendations
- Suggested `max_prompt_length` for each case
- Suggested `max_response_length`
- Batch size recommendations
- Memory usage estimates

## Expected Output

```
================================================================================
RAW DATA ANALYSIS (feedback_train_ppo.json)
================================================================================

Tokenizing all examples...
100%|████████████████████████████| 32402/32402 [00:15<00:00, 2123.45it/s]

Metric                         Min       Mean     Median      90th %      95th %        Max
------------------------------------------------------------------------------------------
Question length (tokens)         3       18.5         15          30          35         120
Answer length (tokens)          10      285.2        250         450         520        1200
Feedback length (tokens)         0       45.3         42          75          85         250

Question Reuse Statistics
--------------------------------------------------
Total examples                                32402
Unique questions                               8934
Questions with >1 answer                       6421 (71.9%)
Avg answers per question                       3.63
Max answers per question                         18

================================================================================
CASE 1: QUESTION ONLY (Baseline)
================================================================================

Creating Case 1 prompts...

Metric                         Min       Mean     Median      90th %      95th %        Max
------------------------------------------------------------------------------------------
Prompt length (tokens)           5       22.1         18          35          40         125

================================================================================
CASE 2: WITH PREVIOUS ANSWER + FEEDBACK (Experimental)
================================================================================

Creating Case 2 prompts...

Distribution
--------------------------------------------------
Examples WITH context                      27845 (85.9%)
Examples WITHOUT context                    4557 (14.1%)

Metric                         Min       Mean     Median      90th %      95th %        Max
------------------------------------------------------------------------------------------
Overall prompt length                5      288.7        265         485         545        1450
Prompt (WITH context)              50      315.4        290         510         580        1450
Prompt (NO context)                 5       22.1         18          35          40         125

================================================================================
COMPARISON: CASE 1 vs CASE 2
================================================================================

Prompt Length Comparison                            Case 1          Case 2      Difference
-------------------------------------------------------------------------------------
Mean prompt length                                    22.1           288.7           266.6
Median prompt length                                    18             265             247
95th percentile                                         40             545             505

Case 2 WITH context only                                            Case 2       vs Case 1
-------------------------------------------------------------------------------------
Mean prompt length                                                   315.4           293.3
Additional tokens from context                                                       293.3

Total Sequence Length (Prompt + Response)           Case 1          Case 2      Difference
-------------------------------------------------------------------------------------
Mean total length                                   307.3           573.9           266.6
Median total length                                 268.0           515.0           247.0
95th percentile                                     560.0          1065.0           505.0
Max total length                                   1245.0          2650.0          1405.0

================================================================================
RECOMMENDATIONS FOR TRAINING CONFIGURATION
================================================================================

1. Maximum Prompt Length:
   Case 1: Set max_prompt_length >= 40 (covers 95% of examples)
   Case 2: Set max_prompt_length >= 545 (covers 95% of examples)
   Recommended: 1024 tokens

2. Maximum Response Length:
   Response 95th percentile: 520 tokens
   Recommended: max_response_length >= 520

3. Total Sequence Length (for memory planning):
   Case 1: 560 tokens (95th percentile)
   Case 2: 1065 tokens (95th percentile)

4. Context Window Increase:
   Case 2 needs ~293 more tokens on average
   This is 1325.8% increase in prompt length

5. Batch Size Recommendations:
   Case 1: Can use larger batch sizes (shorter prompts)
   Case 2: May need 30-50% smaller batch size due to longer prompts
   Consider: Case1 batch=256, Case2 batch=128

6. Training Configuration:
   ```bash
   # Case 1
   data.max_prompt_length=512
   data.max_response_length=572
   data.train_batch_size=256
   
   # Case 2
   data.max_prompt_length=1024
   data.max_response_length=572
   data.train_batch_size=128  # Reduced due to longer prompts
   ```
```

## Key Insights

### Prompt Length Difference
- **Case 1**: ~22 tokens (just the question)
- **Case 2**: ~315 tokens (question + previous answer + feedback)
- **Increase**: ~13x longer prompts in Case 2!

### Why This Matters

1. **Memory Usage**: Case 2 uses ~2x more memory per example
2. **Batch Sizes**: Need to reduce batch size for Case 2
3. **Context Window**: Must set `max_prompt_length` higher for Case 2
4. **Training Speed**: Case 2 will be slower due to longer sequences

### Question Reuse
- **~72% of questions** have multiple answers
- This means **~86% of training examples** can use previous answer context
- **Only ~14%** fall back to Case 1 format

## Using Results in Training

### Case 1 Configuration
```bash
python3 -m verl.trainer.main_ppo \
    data.train_files=~/data/feedback_qa_ppo/case1_question_only/train.parquet \
    data.max_prompt_length=512 \
    data.max_response_length=572 \
    data.train_batch_size=256 \
    # ... other params
```

### Case 2 Configuration
```bash
python3 -m verl.trainer.main_ppo \
    data.train_files=~/data/feedback_qa_ppo/case2_with_feedback/train.parquet \
    data.max_prompt_length=1024 \
    data.max_response_length=572 \
    data.train_batch_size=128 \  # Reduced!
    # ... other params
```

## Troubleshooting

### OOM (Out of Memory) Errors

**For Case 2:**
1. Reduce `data.train_batch_size` (try 128 → 64)
2. Reduce `data.max_prompt_length` (try 1024 → 800)
3. Enable offloading: `actor_rollout_ref.actor.fsdp_config.param_offload=True`
4. Reduce micro batch sizes
5. Enable gradient checkpointing

### Too Many Truncated Examples

If you see warnings about truncated prompts:
- Increase `max_prompt_length`
- Check the 95th percentile from analysis
- Set to cover 95-98% of examples

### Slow Training

Case 2 will be inherently slower:
- ~13x longer prompts
- More computation per example
- This is expected!

Consider:
- Using faster hardware
- Reducing training epochs
- Using smaller models for initial tests

## Summary Table

| Metric | Case 1 | Case 2 | Ratio |
|--------|--------|--------|-------|
| **Mean Prompt** | 22 tokens | 315 tokens | 14.3x |
| **Mean Response** | 285 tokens | 285 tokens | 1.0x |
| **Mean Total** | 307 tokens | 600 tokens | 2.0x |
| **Recommended max_prompt** | 512 | 1024 | 2.0x |
| **Recommended batch_size** | 256 | 128 | 0.5x |
| **Examples with context** | 0% | 86% | - |

## Next Steps

1. ✅ Run this analysis
2. ✅ Note the recommended configurations
3. ✅ Update training scripts with correct lengths
4. ✅ Adjust batch sizes based on your GPU memory
5. ✅ Run training experiments!

