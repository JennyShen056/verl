# Prompt and Response Length Analysis Results

**Generated**: Analysis of `feedback_train_ppo.json` (5,400 examples)  
**Tokenizer**: `meta-llama/Llama-3.1-8B-Instruct`

---

## Executive Summary

### Key Findings

| Metric | Case 1 (Question Only) | Case 2 (With Feedback) | Difference |
|--------|------------------------|------------------------|------------|
| **Mean Prompt Length** | 15.4 tokens | 288.6 tokens | **+273.2 tokens (18.7x)** |
| **Median Prompt Length** | 15 tokens | 232 tokens | **+217 tokens (15.5x)** |
| **95th Percentile Prompt** | 24 tokens | 635 tokens | **+611 tokens (26.5x)** |
| **Mean Total Sequence** | 211.7 tokens | 484.9 tokens | **+273.2 tokens (2.3x)** |
| **95th Percentile Total** | 531 tokens | 1,111 tokens | **+580 tokens (2.1x)** |
| **Max Total Sequence** | 1,970 tokens | 3,137 tokens | **+1,167 tokens (1.6x)** |

---

## Raw Data Statistics

### Question & Answer Characteristics

```
Metric                          Min      Mean    Median    90th %    95th %      Max
─────────────────────────────────────────────────────────────────────────────────────
Question length (tokens)          5      14.4        14        21        23        36
Answer length (tokens)            2     196.3       144       388       514     1,957
Feedback length (tokens)          6      25.0        22        41        47       101
```

### Question Reuse (Critical for Case 2!)

```
Total examples:                    5,400
Unique questions:                  1,564
Questions with >1 answer:          1,564 (100.0%) ✅
Avg answers per question:          3.45
Max answers per question:          6
```

**🎉 Perfect for Case 2!** Every question has multiple answers, meaning 100% of training examples can use the feedback-augmented format!

---

## Case 1: Question Only (Baseline)

### Prompt Format
```
User: How do I get help finding a job?
```

### Statistics
```
Metric                          Min      Mean    Median    90th %    95th %      Max
─────────────────────────────────────────────────────────────────────────────────────
Prompt length (tokens)            6      15.4        15        22        24        37
```

### Characteristics
- ✅ Very short prompts (15 tokens average)
- ✅ Low memory footprint
- ✅ Can use large batch sizes
- ✅ Fast training

---

## Case 2: With Previous Answer + Feedback (Experimental)

### Prompt Format
```
User: Here is a previous answer to this question with feedback:

Question: How do I get help finding a job?

Previous Answer: In this rapidly changing jobs market, it can be difficult to 
know where to start when looking for employment. The good news is that there are 
many resources available to help you in your job search... [~200 tokens]

Feedback: A link to a job search website is included... [~25 tokens]

Now, please answer the same question:

Question: How do I get help finding a job?
```

### Statistics
```
Distribution:
  Examples WITH context:              5,400 (100.0%) ✅
  Examples WITHOUT context:               0 (0.0%)

Metric                          Min      Mean    Median    90th %    95th %      Max
─────────────────────────────────────────────────────────────────────────────────────
Overall prompt length                66     288.6       232       502       635     1,613
Prompt (WITH context)                66     288.6       232       502       635     1,613
```

### Characteristics
- ⚠️ Much longer prompts (288 tokens average, **18.7x longer than Case 1**)
- ⚠️ Higher memory usage (~2x more per example)
- ⚠️ Requires reduced batch size
- ⚠️ Slower training
- ✅ 100% of examples get feedback context!

---

## Detailed Comparison

### Prompt Length Distribution

```
                        Case 1          Case 2       Difference
────────────────────────────────────────────────────────────────
Mean                      15.4          288.6          +273.2
Median                      15            232           +217
90th percentile             22            502           +480
95th percentile             24            635           +611
Max                         37          1,613         +1,576

Increase Factor:         1.0x          18.7x
```

### Total Sequence Length (Prompt + Response)

```
                        Case 1          Case 2       Difference
────────────────────────────────────────────────────────────────
Mean                     211.7          484.9          +273.2
Median                     159            387           +228
95th percentile            531          1,111           +580
Max                      1,970          3,137         +1,167
```

### Context Window Increase

**Case 2 adds ~273 tokens per prompt on average**
- This is a **1,775% increase** in prompt length!
- Almost **19x longer** than Case 1
- The extra tokens come from:
  - Previous answer: ~196 tokens (mean)
  - Feedback: ~25 tokens (mean)
  - Formatting text: ~52 tokens

---

## Training Configuration Recommendations

### Case 1: Question Only

```bash
# Recommended configuration for Case 1
python3 -m verl.trainer.main_ppo \
    data.train_files=~/data/feedback_qa_ppo/case1_question_only/train.parquet \
    data.val_files=~/data/feedback_qa_ppo/case1_question_only/valid.parquet \
    data.max_prompt_length=512 \
    data.max_response_length=565 \
    data.train_batch_size=256 \
    # ... other parameters
```

**Rationale:**
- `max_prompt_length=512`: Covers 100% of prompts (max is 37 tokens)
- `max_response_length=565`: Covers 95% of responses (95th percentile = 514)
- `train_batch_size=256`: Can use large batches due to short prompts

### Case 2: With Previous Answer + Feedback

```bash
# Recommended configuration for Case 2
python3 -m verl.trainer.main_ppo \
    data.train_files=~/data/feedback_qa_ppo/case2_with_feedback/train.parquet \
    data.val_files=~/data/feedback_qa_ppo/case2_with_feedback/valid.parquet \
    data.max_prompt_length=1024 \
    data.max_response_length=565 \
    data.train_batch_size=128 \
    # ... other parameters
```

**Rationale:**
- `max_prompt_length=1024`: Covers 95% of prompts (95th percentile = 635)
- `max_response_length=565`: Same as Case 1 (responses are identical)
- `train_batch_size=128`: **REDUCED by 50%** due to 2x longer sequences

---

## Memory and Performance Implications

### Memory Usage per Example

| Component | Case 1 | Case 2 | Ratio |
|-----------|--------|--------|-------|
| Prompt | 15 tokens | 289 tokens | 19.3x |
| Response | 196 tokens | 196 tokens | 1.0x |
| **Total** | **212 tokens** | **485 tokens** | **2.3x** |

### GPU Memory Scaling

Assuming 8xA100 (80GB each):

**Case 1:**
- Max sequence: ~531 tokens (95th percentile)
- Can fit: batch_size=256 easily
- Total throughput: ~2,048 sequences/step (with 8 GPUs)

**Case 2:**
- Max sequence: ~1,111 tokens (95th percentile)
- Can fit: batch_size=128 (50% reduction)
- Total throughput: ~1,024 sequences/step (with 8 GPUs)

**Result:** Case 2 has ~50% training throughput of Case 1

### Training Speed

| Metric | Case 1 | Case 2 | Impact |
|--------|--------|--------|--------|
| Tokens/example | 212 | 485 | 2.3x more |
| Batch size | 256 | 128 | 0.5x less |
| Sequences/step | 2,048 | 1,024 | 0.5x less |
| **Relative speed** | **1.0x** | **~0.43x** | **2.3x slower** |

**Case 2 will take approximately 2.3x longer to train** than Case 1 for the same number of examples.

---

## Recommendations for Experimentation

### Initial Tests (Small Scale)

1. **Start with smaller models first**
   - Test with 1B-7B models before scaling to 70B
   - Verify prompt formats are correct
   - Check that RM scores make sense

2. **Use subset of data**
   - Train on 10-20% of data first
   - Validate the training pipeline
   - Ensure convergence

### Production Training (Full Scale)

3. **Hardware configuration**
   - Case 1: 8xA100 80GB, batch_size=256 per GPU → 2,048 global
   - Case 2: 8xA100 80GB, batch_size=128 per GPU → 1,024 global
   
4. **Memory optimizations for Case 2**
   - Enable FSDP offloading: `actor_rollout_ref.actor.fsdp_config.param_offload=True`
   - Enable gradient checkpointing
   - Use FlashAttention-2
   - Consider mixed precision (fp16 or bf16)

5. **Monitoring**
   - Track prompt truncation rates
   - Monitor memory usage per GPU
   - Watch for OOM errors
   - Compare convergence speed Case 1 vs Case 2

### Troubleshooting OOM

If Case 2 runs out of memory:

```bash
# Option 1: Reduce batch size further
data.train_batch_size=64  # or even 32

# Option 2: Reduce max_prompt_length
data.max_prompt_length=800  # Covers ~90% instead of 95%

# Option 3: Enable aggressive offloading
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.cpu_offload=True

# Option 4: Use smaller max_response_length
data.max_response_length=400  # Covers ~85% of responses
```

---

## Key Insights for Your Experiment

### ✅ Positive Findings

1. **100% context coverage**: All questions have multiple answers!
   - Every training example in Case 2 will have feedback context
   - No fallback to Case 1 format needed

2. **Reasonable prompt lengths**: 95th percentile is 635 tokens
   - Fits comfortably in 1024 token context window
   - Modern LLMs handle this easily

3. **Consistent response lengths**: Same distribution for both cases
   - Fair comparison between Case 1 and Case 2
   - RM will score equivalent outputs

### ⚠️ Challenges to Consider

1. **Significant memory increase**: 2.3x more tokens per example
   - Need to reduce batch sizes
   - Training will be slower

2. **Long tail of prompts**: Some reach 1,613 tokens
   - Consider truncating extreme outliers
   - Or increase max_prompt_length to 1536/2048

3. **Training cost**: Case 2 is ~2.3x slower than Case 1
   - Budget more GPU hours
   - Or reduce number of epochs

### 🎯 Experimental Validity

The setup is well-designed for a fair comparison:
- ✅ Same response lengths (same answers)
- ✅ Same RM scoring (Q+A format)
- ✅ Only difference: feedback context in training
- ✅ Inference is identical (question only → answer)
- ✅ Can directly attribute performance differences to feedback

---

## Quick Reference Table

| Configuration Parameter | Case 1 | Case 2 | Notes |
|------------------------|--------|--------|-------|
| `max_prompt_length` | 512 | 1024 | 2x for Case 2 |
| `max_response_length` | 565 | 565 | Same for both |
| `train_batch_size` | 256 | 128 | 50% smaller for Case 2 |
| Expected training speed | 1.0x | 0.43x | Case 2 is slower |
| Memory per example | 212 tokens | 485 tokens | 2.3x for Case 2 |
| Context coverage | 100% | 100% | Both cover all data |

---

## Next Steps

1. ✅ Analysis complete - review results above
2. ✅ Update training scripts with recommended parameters
3. ⏭️ Run preprocessing for both cases
4. ⏭️ Train Case 1 (baseline) first
5. ⏭️ Train Case 2 (experimental) 
6. ⏭️ Compare results and see if feedback helps!

---

**Questions or issues?** Refer to:
- `LENGTH_ANALYSIS_README.md` - How to run analysis
- `EXPERIMENTAL_DESIGN.md` - Overall experiment design
- `RUN_EXPERIMENT.md` - Step-by-step execution guide

