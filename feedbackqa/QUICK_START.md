# Quick Start Guide: Feedback-Augmented Learning Experiment

## TL;DR

Test if showing models **feedback examples** improves answer quality.

**Case 1 (Baseline)**: Question → Answer  
**Case 2 (Experimental)**: Question + Example Q+A+Feedback → Answer

Both scored by same RM. Winner = better answers.

---

## 5-Step Quick Start

### 1️⃣ Train Reward Model (~3 hours)
```bash
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --output_dir ./feedback_qa_reward_model
```

### 2️⃣ Preprocess Data (~10 minutes)
```bash
# Case 1: Question only
python feedbackqa/preprocess_ppo_case1_question_only.py

# Case 2: With feedback
python feedbackqa/preprocess_ppo_case2_with_feedback.py
```

### 3️⃣ Train Case 1 (~12 hours)
```bash
bash feedbackqa/run_ppo_case1_question_only.sh
```

### 4️⃣ Train Case 2 (~12 hours)
```bash
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

### 5️⃣ Compare Results
Check W&B: `feedback_qa_experiment`
- `case1_question_only_baseline`
- `case2_with_feedback_experimental`

Compare `actor/reward` and validation metrics.

---

## What Each Script Does

| Script | Purpose | Key Feature |
|--------|---------|-------------|
| `rm_train.py` | Train reward model | verl-compatible (TokenClassification, num_labels=1) |
| `preprocess_ppo_case1_*` | Create baseline data | Input: just question |
| `preprocess_ppo_case2_*` | Create experimental data | Input: question + random example + feedback |
| `run_ppo_case1_*` | Train baseline model | Standard PPO |
| `run_ppo_case2_*` | Train experimental model | PPO with feedback context |

---

## Key Design Choices

### Why Random Examples in Case 2?
- **Prevents memorization**: Different example each time
- **Diverse learning**: See various quality patterns
- **Generalizes better**: Not tied to specific feedback

### Why RM Scores Generated Answer Only?
- **Fair comparison**: Both cases judged equally
- **No gaming**: Can't just copy example
- **Tests generalization**: Did model learn the principle?

### Why Same Test Format?
- **Fair evaluation**: Both get just questions
- **Real-world**: No feedback at inference
- **Pure comparison**: Which learned better?

---

## Expected Results

### If Case 2 Wins 🎉
- Feedback examples help learning
- Models internalize quality criteria
- **Use feedback-augmented training!**

### If Case 1 Wins 🤔
- Direct RL is sufficient
- Examples may confuse
- **Stick with question-only!**

### If Tie 🤝
- Feedback doesn't add value
- RM scores alone sufficient
- **Either approach works!**

---

## Data Flow Diagram

```
Training Data
     ↓
Train Reward Model (RM)
     ↓
RM learns to score answers (0/1)
     ↓
     ┌─────────────────┴─────────────────┐
     │                                   │
  Case 1                              Case 2
  ├─ Input: Q                         ├─ Input: Single prompt with
  │                                   │   Example Q+A+Feedback + Q
  ├─ Generate: A                      ├─ Generate: A
  ├─ RM scores: Q+A                   ├─ RM scores: Original Q+A
  └─ Learn from reward                └─ Learn from reward + examples
     │                                   │
     ↓                                   ↓
  Model A                             Model B
     │                                   │
     └─────────────────┬─────────────────┘
                       ↓
                Test (both get Q only)
                       ↓
                   Compare!
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM | Reduce batch sizes, enable offloading |
| RM not loading | Run `verify_trained_model.py` |
| Poor performance | Check RM quality, adjust LR |
| Slow training | Use more GPUs, reduce sequence lengths |

---

## Files Created

### Core Scripts
- ✅ `rm_train.py` - Modified for verl
- ✅ `preprocess_ppo_case1_question_only.py`
- ✅ `preprocess_ppo_case2_with_feedback.py`
- ✅ `run_ppo_case1_question_only.sh`
- ✅ `run_ppo_case2_with_feedback.sh`

### Documentation
- ✅ `EXPERIMENTAL_DESIGN.md` - Full design details
- ✅ `RUN_EXPERIMENT.md` - Step-by-step guide
- ✅ `QUICK_START.md` - This file
- ✅ `CHANGES_SUMMARY.md` - RM modifications
- ✅ `README_VERL_INTEGRATION.md` - verl integration

### Utilities
- ✅ `verify_trained_model.py` - Check RM compatibility

---

## Timeline

| Task | Time | When |
|------|------|------|
| RM Training | 3 hours | Day 1 morning |
| Data Prep | 10 min | Day 1 afternoon |
| Case 1 Training | 12 hours | Day 1 overnight |
| Case 2 Training | 12 hours | Day 2 overnight |
| Evaluation | 2 hours | Day 3 morning |
| Analysis | 2 hours | Day 3 afternoon |

**Total**: 2-3 days

---

## Success Criteria

Your experiment is successful if you can answer:

1. ✅ Which case achieved higher rewards?
2. ✅ Which converged faster?
3. ✅ Which is more stable?
4. ✅ Why did one outperform the other?

---

## One-Line Summary

**Test if showing models feedback examples during training improves their answer quality compared to learning from reward signals alone.**

🚀 **Ready to start? Run Step 1!**

