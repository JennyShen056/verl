#!/usr/bin/env python3
"""
Complete Experimental Pipeline Summary
=======================================

End-to-end guide for the feedback-augmented learning experiment.
"""

# 🎯 **Experiment Overview**

**Research Question**: Does training with feedback-augmented examples help models generate better answers?

**Hypothesis**: Case 2 (feedback-augmented) > Case 1 (baseline)

**Approach**:
- **Case 1 (Baseline)**: Train PPO with question-only prompts
- **Case 2 (Experimental)**: Train PPO with feedback-augmented prompts (previous answer + feedback)
- **Evaluation**: Compare using reward model scores

---

# 🚀 **Complete Workflow**

## **Phase 1: Data Preparation** ✅

### 1.1 Preprocess Data

```bash
# Case 1: Question Only
python feedbackqa/preprocess_ppo_case1_question_only.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --test_file feedbackqa/feedback_test_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case1_question_only

# Case 2: With Feedback
python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --test_file feedbackqa/feedback_test_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case2_with_feedback
```

**Output:**
- `train.parquet`, `valid.parquet`, `test.parquet` for each case
- Different prompt formats (see `FORMAT_COMPARISON.md`)

---

## **Phase 2: Training** ✅

### 2.1 Train Case 1 (Baseline)

```bash
bash feedbackqa/run_ppo_case1_question_only.sh
```

**Configuration:**
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Reward Model: `sfairXC/FsfairX-LLaMA3-RM-v0.1`
- Batch size: 256
- Max prompt: 512 tokens
- Max response: 565 tokens
- Epochs: 1
- Time: ~30-60 min/epoch

**Checkpoints saved to:**
```
checkpoints/feedback_qa_experiment/case1_question_only_baseline/global_step_*/
```

### 2.2 Train Case 2 (Experimental)

```bash
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

**Configuration:**
- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Reward Model: `sfairXC/FsfairX-LLaMA3-RM-v0.1`
- Batch size: 128 (reduced for longer prompts)
- Max prompt: 1024 tokens
- Max response: 565 tokens
- Epochs: 10
- Time: ~90-150 min/epoch

**Checkpoints saved to:**
```
checkpoints/feedback_qa_experiment/case2_with_feedback_experimental/global_step_*/
```

### 2.3 Monitor Training

**Weights & Biases:**
- Project: `feedback_qa_experiment`
- Runs: 
  - `case1_question_only_baseline`
  - `case2_with_feedback_experimental`

**Key metrics to watch:**
- Average reward (should increase)
- Training loss (should decrease)
- Validation metrics (every 2 steps)

---

## **Phase 3: Evaluation** ✅

### 3.1 Quick Evaluation (Automated)

```bash
# Evaluate at global_step 1
bash feedbackqa/run_full_evaluation.sh 1
```

This runs the complete pipeline:
1. Merges checkpoints to HuggingFace format
2. Runs inference on test set
3. Evaluates with reward model
4. Compares results statistically

**Time:** ~1-2 hours total

### 3.2 Manual Evaluation (Step by Step)

If you prefer manual control:

```bash
# Step 1: Merge checkpoints
bash feedbackqa/merge_checkpoint.sh case1_question_only_baseline 1
bash feedbackqa/merge_checkpoint.sh case2_with_feedback_experimental 1

# Step 2: Run inference
python feedbackqa/inference.py \
    --model_path checkpoints/feedback_qa_experiment/case1_question_only_baseline/global_step_1/actor/huggingface \
    --test_file feedbackqa/feedback_test_ppo.json \
    --output_file outputs/case1_predictions.json

python feedbackqa/inference.py \
    --model_path checkpoints/feedback_qa_experiment/case2_with_feedback_experimental/global_step_1/actor/huggingface \
    --test_file feedbackqa/feedback_test_ppo.json \
    --output_file outputs/case2_predictions.json

# Step 3: Evaluate with reward model
python feedbackqa/evaluate.py \
    --predictions_file outputs/case1_predictions.json \
    --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --output_file outputs/case1_evaluation.json

python feedbackqa/evaluate.py \
    --predictions_file outputs/case2_predictions.json \
    --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --output_file outputs/case2_evaluation.json

# Step 4: Compare results
python feedbackqa/compare_results.py \
    --case1_file outputs/case1_evaluation.json \
    --case2_file outputs/case2_evaluation.json \
    --output_file outputs/comparison_report.json
```

---

# 📊 **Expected Results**

## **If Hypothesis is Confirmed** ✅

```
🎯 Conclusion:
✅ HYPOTHESIS CONFIRMED!
   Case 2 (Feedback-Augmented) significantly outperforms Case 1 (Baseline)
   Mean reward improvement: 0.1234 (15.6%)
   Effect size: medium
   p-value: 0.0012

   💡 Conclusion: Training with feedback-augmented examples HELPS models learn!
```

**Interpretation:**
- Providing feedback context during training improves answer quality
- Model learns from example answers and feedback
- Statistically significant with medium/large effect size

## **If Hypothesis is Rejected** ❌

```
🎯 Conclusion:
❌ HYPOTHESIS REJECTED!
   Case 1 (Baseline) significantly outperforms Case 2 (Feedback-Augmented)
   
   💡 Conclusion: Feedback-augmented training did NOT help in this experiment.
```

**Possible reasons:**
- Feedback examples may confuse the model
- Simple questions don't need feedback context
- Training epochs may need adjustment

## **If No Significant Difference** ⚠️

```
🎯 Conclusion:
⚠️  NO SIGNIFICANT DIFFERENCE
   p-value: 0.3456 (> 0.05)
   
   💡 Conclusion: Feedback-augmented training shows no significant impact.
```

**Possible reasons:**
- Effect exists but too small to detect
- Need more training data
- Need longer training
- Reward model not sensitive enough

---

# 📁 **File Structure**

```
feedbackqa/
├── Data Files
│   ├── feedback_train_ppo.json         # Training data
│   ├── feedback_valid_ppo.json         # Validation data
│   └── feedback_test_ppo.json          # Test data
│
├── Preprocessing Scripts
│   ├── preprocess_ppo_case1_question_only.py
│   └── preprocess_ppo_case2_with_feedback.py
│
├── Training Scripts
│   ├── run_ppo_case1_question_only.sh
│   └── run_ppo_case2_with_feedback.sh
│
├── Evaluation Scripts
│   ├── merge_checkpoint.sh             # Convert FSDP to HuggingFace
│   ├── inference.py                    # Generate answers
│   ├── evaluate.py                     # Score with reward model
│   ├── compare_results.py              # Statistical comparison
│   └── run_full_evaluation.sh          # Automated pipeline
│
├── Analysis Scripts
│   └── analyze_prompt_response_lengths.py
│
└── Documentation
    ├── EXPERIMENTAL_DESIGN.md          # Experiment overview
    ├── TRAINING_WORKFLOW.md            # Training guide
    ├── ALL_FIXES_SUMMARY.md            # Bug fixes
    ├── ANALYSIS_RESULTS.md             # Length analysis
    ├── INFERENCE_AND_EVALUATION_GUIDE.md
    └── THIS_FILE.md                    # Complete summary
```

```
outputs/
├── case1_predictions.json              # Case 1 generated answers
├── case1_evaluation.json               # Case 1 with scores
├── case2_predictions.json              # Case 2 generated answers
├── case2_evaluation.json               # Case 2 with scores
└── comparison_report.json              # Statistical results
```

```
checkpoints/
└── feedback_qa_experiment/
    ├── case1_question_only_baseline/
    │   └── global_step_*/
    │       └── actor/
    │           └── huggingface/        # Merged model
    └── case2_with_feedback_experimental/
        └── global_step_*/
            └── actor/
                └── huggingface/        # Merged model
```

---

# 🛠️ **Created Scripts and Tools**

## **Training**
1. ✅ `run_ppo_case1_question_only.sh` - Train baseline model
2. ✅ `run_ppo_case2_with_feedback.sh` - Train experimental model

## **Evaluation**
3. ✅ `merge_checkpoint.sh` - Convert checkpoints
4. ✅ `inference.py` - Generate predictions
5. ✅ `evaluate.py` - Score with reward model
6. ✅ `compare_results.py` - Statistical comparison
7. ✅ `run_full_evaluation.sh` - Automated pipeline

## **Analysis**
8. ✅ `analyze_prompt_response_lengths.py` - Token analysis

## **Documentation**
9. ✅ `EXPERIMENTAL_DESIGN.md` - Experiment design
10. ✅ `TRAINING_WORKFLOW.md` - Training guide
11. ✅ `ALL_FIXES_SUMMARY.md` - Bug fixes
12. ✅ `ANALYSIS_RESULTS.md` - Length analysis results
13. ✅ `INFERENCE_AND_EVALUATION_GUIDE.md` - Evaluation guide
14. ✅ `COMPLETE_PIPELINE_SUMMARY.md` - This file

---

# ✅ **Verification Checklist**

Before running:

## **Data Preparation**
- [ ] Test data exists: `feedbackqa/feedback_test_ppo.json`
- [ ] Preprocessing complete for Case 1
- [ ] Preprocessing complete for Case 2

## **Training**
- [ ] Case 1 training complete
- [ ] Case 2 training complete
- [ ] Checkpoints saved properly
- [ ] W&B logs show convergence

## **Evaluation**
- [ ] Checkpoint merge successful
- [ ] Inference runs without errors
- [ ] Reward model evaluation works
- [ ] Comparison produces results

## **Results**
- [ ] Statistical significance determined
- [ ] Effect size calculated
- [ ] Hypothesis confirmed/rejected/inconclusive
- [ ] Sample predictions reviewed manually

---

# 🎓 **Key Insights**

## **Training Insights**

1. **Case 2 trains slower**: 2.3x more tokens per example
   - Expected: ~90-150 min/epoch vs 30-60 min/epoch

2. **Both use same reward model**: Fair comparison
   - RM only sees Question + Answer (not the feedback context)

3. **Validation during training**: Monitor convergence
   - Check W&B for reward trends

## **Evaluation Insights**

1. **Reward model scoring**: Both cases scored identically
   - Only the generated answer is evaluated
   - Training context doesn't affect evaluation

2. **Statistical tests**: Multiple tests for robustness
   - T-test: Parametric
   - Mann-Whitney U: Non-parametric
   - Effect size: Practical significance

3. **Manual inspection important**: Numbers don't tell full story
   - Read generated answers
   - Look for quality patterns
   - Understand failure modes

---

# 🚨 **Common Issues**

## **Training Issues**
- OOM → Reduce batch size
- Slow convergence → Increase epochs
- Unstable rewards → Adjust learning rate

## **Evaluation Issues**
- Checkpoint not found → Check training logs
- OOM during inference → Reduce batch size
- Poor generations → Try later checkpoint

## **Statistical Issues**
- No significance → Need more data or training
- High variance → Model inconsistent
- Unexpected direction → Check for bugs

---

# 📚 **Quick Reference**

## **Run Everything (After Training)**

```bash
# Complete evaluation pipeline
bash feedbackqa/run_full_evaluation.sh 1

# Check results
cat outputs/comparison_report.json
```

## **Check Training Status**

```bash
# List checkpoints
ls -d checkpoints/feedback_qa_experiment/*/global_step_*

# Check W&B
# Navigate to: https://wandb.ai/your-username/feedback_qa_experiment
```

## **Manual Inspection**

```python
# Load and inspect predictions
import json

with open('outputs/case1_predictions.json') as f:
    case1 = json.load(f)

with open('outputs/case2_predictions.json') as f:
    case2 = json.load(f)

# Compare same question
for i in range(5):
    print(f"\n{'='*60}")
    print(f"Question {i+1}: {case1[i]['question']}")
    print(f"\nCase 1 Answer: {case1[i]['generated_answer']}")
    print(f"\nCase 2 Answer: {case2[i]['generated_answer']}")
    print(f"\nGround Truth: {case1[i]['ground_truth']}")
```

---

# 🎯 **Success Criteria**

Your experiment is successful if:

1. ✅ **Both models train successfully**
   - Checkpoints saved
   - Rewards increase during training
   - No crashes or errors

2. ✅ **Evaluation completes**
   - Inference generates answers
   - Reward model scores computed
   - Statistical comparison done

3. ✅ **Results are interpretable**
   - Clear conclusion (confirmed/rejected/inconclusive)
   - Statistical significance determined
   - Effect size calculated

4. ✅ **Insights gained**
   - Understand if feedback helps
   - Know which approach is better
   - Have data to support findings

---

# 🚀 **Next Steps**

After completing the experiment:

1. **Document findings**
   - Write up results
   - Include statistical tests
   - Show sample predictions

2. **Share results**
   - Present to team
   - Write blog post
   - Submit paper

3. **Iterate**
   - Try different feedback formats
   - Test on other datasets
   - Experiment with different models

4. **Deploy**
   - Use best model in production
   - Monitor performance
   - Collect real-world feedback

---

**Status:** 🟢 **Complete pipeline ready!** 🎉

You have:
- ✅ Training scripts (fixed and tested)
- ✅ Evaluation pipeline (automated)
- ✅ Analysis tools (comprehensive)
- ✅ Documentation (detailed)

**Now go run your experiment and discover if feedback helps!** 🚀

