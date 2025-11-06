#!/usr/bin/env python3
"""
Inference and Evaluation Guide
===============================

Complete guide to evaluating your trained PPO models.
"""

# 🎯 **Quick Start**

After training is complete, run the full evaluation pipeline:

```bash
bash feedbackqa/run_full_evaluation.sh 1
```

This will:
1. Merge checkpoints to HuggingFace format
2. Run inference on test set
3. Evaluate using reward model
4. Compare Case 1 vs Case 2 statistically

---

# 📋 **Step-by-Step Guide**

## **Prerequisites**

Ensure you have:
- ✅ Trained Case 1 model (checkpoints saved)
- ✅ Trained Case 2 model (checkpoints saved)
- ✅ Test data (`feedbackqa/feedback_test_ppo.json`)
- ✅ Required packages: `transformers`, `torch`, `scipy`, `numpy`

---

## **Step 1: Merge Checkpoints**

Convert FSDP checkpoints to HuggingFace format for inference.

### **Case 1 (Baseline):**
```bash
bash feedbackqa/merge_checkpoint.sh case1_question_only_baseline 1
```

### **Case 2 (Experimental):**
```bash
bash feedbackqa/merge_checkpoint.sh case2_with_feedback_experimental 1
```

**What this does:**
- Reads FSDP checkpoint from `checkpoints/feedback_qa_experiment/<case_name>/global_step_1/actor/`
- Converts to HuggingFace format
- Saves to `.../actor/huggingface/`

**Check available checkpoints:**
```bash
ls -d checkpoints/feedback_qa_experiment/*/global_step_*
```

**Expected output:**
```
checkpoints/feedback_qa_experiment/case1_question_only_baseline/global_step_1/
checkpoints/feedback_qa_experiment/case2_with_feedback_experimental/global_step_1/
...
```

---

## **Step 2: Run Inference**

Generate answers on the test set.

### **Case 1:**
```bash
python feedbackqa/inference.py \
    --model_path checkpoints/feedback_qa_experiment/case1_question_only_baseline/global_step_1/actor/huggingface \
    --test_file feedbackqa/feedback_test_ppo.json \
    --output_file outputs/case1_predictions.json \
    --batch_size 8 \
    --max_new_tokens 512
```

### **Case 2:**
```bash
python feedbackqa/inference.py \
    --model_path checkpoints/feedback_qa_experiment/case2_with_feedback_experimental/global_step_1/actor/huggingface \
    --test_file feedbackqa/feedback_test_ppo.json \
    --output_file outputs/case2_predictions.json \
    --batch_size 8 \
    --max_new_tokens 512
```

**Parameters:**
- `--model_path`: Path to merged HuggingFace model
- `--test_file`: Test data JSON file
- `--output_file`: Where to save predictions
- `--batch_size`: Batch size for inference (adjust for GPU memory)
- `--max_new_tokens`: Maximum tokens to generate per answer

**Expected time:** ~10-30 minutes depending on test set size and GPU

**Output format:**
```json
[
  {
    "question": "How do I...",
    "ground_truth": "Original answer...",
    "feedback": "Original feedback...",
    "rating": "Good",
    "generated_answer": "Model's generated answer..."
  },
  ...
]
```

---

## **Step 3: Evaluate with Reward Model**

Score generated answers using the reward model.

### **Case 1:**
```bash
python feedbackqa/evaluate.py \
    --predictions_file outputs/case1_predictions.json \
    --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --output_file outputs/case1_evaluation.json \
    --batch_size 8
```

### **Case 2:**
```bash
python feedbackqa/evaluate.py \
    --predictions_file outputs/case2_predictions.json \
    --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --output_file outputs/case2_evaluation.json \
    --batch_size 8
```

**Parameters:**
- `--predictions_file`: Predictions from inference step
- `--reward_model_path`: HuggingFace path or local path to reward model
- `--output_file`: Where to save evaluation results
- `--batch_size`: Batch size for reward model scoring

**Expected time:** ~5-15 minutes

**Output format:**
```json
{
  "metrics": {
    "mean_reward": 0.7234,
    "std_reward": 0.1823,
    "median_reward": 0.7456,
    "min_reward": 0.2145,
    "max_reward": 0.9876,
    "q25_reward": 0.6234,
    "q75_reward": 0.8456,
    "num_samples": 1000
  },
  "predictions": [
    {
      "question": "...",
      "generated_answer": "...",
      "reward_score": 0.8234
    },
    ...
  ]
}
```

---

## **Step 4: Compare Results**

Statistically compare Case 1 vs Case 2.

```bash
python feedbackqa/compare_results.py \
    --case1_file outputs/case1_evaluation.json \
    --case2_file outputs/case2_evaluation.json \
    --output_file outputs/comparison_report.json
```

**What this does:**
- Loads both evaluation results
- Computes metric differences
- Runs statistical tests (t-test, Mann-Whitney U)
- Calculates effect size (Cohen's d)
- Determines if differences are significant

**Expected output:**
```
================================================================================
EXPERIMENT RESULTS: Case 1 (Baseline) vs Case 2 (Feedback-Augmented)
================================================================================

📊 Overall Metrics:
--------------------------------------------------------------------------------
Metric               Case 1       Case 2   Difference   % Change
--------------------------------------------------------------------------------
mean_reward          0.6234       0.7456       0.1222      19.6% ✓
std_reward           0.1823       0.1456      -0.0367     -20.1% ✓
median_reward        0.6345       0.7567       0.1222      19.3% ✓
...

📈 Statistical Tests:
--------------------------------------------------------------------------------
T-test:
  t-statistic: 8.7234
  p-value: 0.0001 ✓ Significant (α=0.05)

Mann-Whitney U test:
  U-statistic: 234567.0
  p-value: 0.0002 ✓ Significant (α=0.05)

Effect Size:
  Cohen's d: 0.6234
  Interpretation: MEDIUM

Pairwise Comparison:
  Case 2 wins: 687 (68.7%)
  Ties: 45
  Case 1 wins: 268 (26.8%)

🎯 Conclusion:
--------------------------------------------------------------------------------
✅ HYPOTHESIS CONFIRMED!
   Case 2 (Feedback-Augmented) significantly outperforms Case 1 (Baseline)
   Mean reward improvement: 0.1222 (19.6%)
   Effect size: medium

   💡 Conclusion: Training with feedback-augmented examples HELPS models learn!
================================================================================
```

---

# 🔄 **Complete Workflow (Automated)**

Run everything with a single command:

```bash
bash feedbackqa/run_full_evaluation.sh 1
```

**Arguments:**
- `1`: Global step number to evaluate (default: 1)

**This automates:**
1. ✅ Merging Case 1 checkpoint
2. ✅ Merging Case 2 checkpoint
3. ✅ Running inference for Case 1
4. ✅ Running inference for Case 2
5. ✅ Evaluating Case 1
6. ✅ Evaluating Case 2
7. ✅ Comparing results

**Total time:** ~1-2 hours depending on test set size

---

# 📊 **Understanding the Results**

## **Metrics Explained**

| Metric | Meaning | Good Value |
|--------|---------|------------|
| `mean_reward` | Average reward score | Higher is better |
| `std_reward` | Variability in scores | Lower = more consistent |
| `median_reward` | Middle score | Higher is better |
| `min/max_reward` | Range of scores | Check for outliers |
| `q25/q75_reward` | 25th/75th percentiles | Distribution shape |

## **Statistical Tests**

### **T-Test**
- Tests if mean rewards differ significantly
- **p < 0.05**: Statistically significant difference
- **p >= 0.05**: No significant difference

### **Mann-Whitney U Test**
- Non-parametric version of t-test
- More robust to outliers
- **p < 0.05**: Significant difference

### **Cohen's d (Effect Size)**
- Measures magnitude of difference
- **< 0.2**: Negligible
- **0.2-0.5**: Small
- **0.5-0.8**: Medium
- **> 0.8**: Large

## **Interpreting Results**

### **✅ Hypothesis Confirmed**
Case 2 > Case 1 with p < 0.05
→ Feedback-augmented training HELPS!

### **❌ Hypothesis Rejected**
Case 1 > Case 2 with p < 0.05
→ Feedback-augmented training HURTS!

### **⚠️ No Significant Difference**
p >= 0.05
→ Feedback has NO clear impact

---

# 🛠️ **Troubleshooting**

## **Issue: Checkpoint not found**

**Error:**
```
ERROR: Checkpoint not found at checkpoints/...
```

**Fix:**
1. Check training completed successfully
2. List available checkpoints:
   ```bash
   ls -R checkpoints/feedback_qa_experiment/
   ```
3. Use correct `global_step` number

## **Issue: Out of memory during inference**

**Fix:**
Reduce batch size:
```bash
python feedbackqa/inference.py \
    --batch_size 4 \  # or 2, or 1
    ...
```

## **Issue: Model generates gibberish**

**Possible causes:**
1. Training didn't converge
2. Wrong checkpoint selected
3. Need more training epochs

**Fix:**
1. Check training logs/W&B
2. Try a later checkpoint (higher global_step)
3. Adjust generation parameters:
   ```bash
   --max_new_tokens 256 \  # shorter responses
   --temperature 0.5 \     # less random (not in current script, would need to add)
   ```

## **Issue: Reward scores all similar**

**Possible causes:**
1. Reward model not discriminative enough
2. All answers genuinely similar quality
3. Need more diverse test set

**Check:**
1. Review sample predictions manually
2. Check reward model metrics during RM training
3. Try different reward model if available

---

# 📁 **Output Files**

After evaluation, you'll have:

```
outputs/
├── case1_predictions.json          # Case 1 generated answers
├── case1_evaluation.json          # Case 1 with reward scores
├── case2_predictions.json          # Case 2 generated answers
├── case2_evaluation.json          # Case 2 with reward scores
└── comparison_report.json          # Statistical comparison
```

**What to review:**

1. **`comparison_report.json`**: 
   - Main results
   - Statistical significance
   - Effect size

2. **`case1_evaluation.json` / `case2_evaluation.json`**:
   - Individual scores
   - Find best/worst examples
   - Understand model behavior

3. **`case1_predictions.json` / `case2_predictions.json`**:
   - Read generated answers
   - Compare quality manually
   - Find patterns

---

# 🎓 **Advanced Usage**

## **Evaluate Multiple Checkpoints**

Compare different training stages:

```bash
# Evaluate step 1 (early)
bash feedbackqa/run_full_evaluation.sh 1

# Evaluate step 5 (middle)
bash feedbackqa/run_full_evaluation.sh 5

# Evaluate step 10 (final)
bash feedbackqa/run_full_evaluation.sh 10
```

## **Use Custom Reward Model**

If you trained your own reward model:

```bash
python feedbackqa/evaluate.py \
    --predictions_file outputs/case1_predictions.json \
    --reward_model_path ./feedback_qa_reward_model/final_model \
    --output_file outputs/case1_evaluation_custom_rm.json
```

## **Evaluate on Subset**

For quick testing:

```python
# Manually edit test file or create subset
import json

with open('feedbackqa/feedback_test_ppo.json') as f:
    data = json.load(f)

# Take first 100 examples
subset = data[:100]

with open('feedbackqa/feedback_test_subset.json', 'w') as f:
    json.dump(subset, f)

# Then run inference on subset
```

---

# 📚 **Related Files**

| File | Purpose |
|------|---------|
| `merge_checkpoint.sh` | Convert FSDP to HuggingFace |
| `inference.py` | Generate answers on test set |
| `evaluate.py` | Score answers with reward model |
| `compare_results.py` | Statistical comparison |
| `run_full_evaluation.sh` | Run complete pipeline |
| THIS FILE | Usage guide |

---

# 🎯 **Summary Checklist**

After training completes:

- [ ] Check training logs/W&B to verify convergence
- [ ] Merge checkpoints for both cases
- [ ] Run inference on test set
- [ ] Evaluate with reward model
- [ ] Compare results statistically
- [ ] Review `comparison_report.json`
- [ ] Manually inspect sample predictions
- [ ] Document findings for your experiment

---

**Status:** 🟢 **Ready to evaluate your trained models!** 🚀

For questions or issues, refer to:
- `ALL_FIXES_SUMMARY.md` - Training fixes
- `TRAINING_WORKFLOW.md` - Training guide
- `EXPERIMENTAL_DESIGN.md` - Experiment overview

