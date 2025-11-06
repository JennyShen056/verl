# Complete Training Workflow

## 📋 Current Status

✅ Code is ready  
✅ Scripts have proper error checking  
✅ Using pre-trained reward model (FsfairX-LLaMA3-RM-v0.1)  
❌ **Need to preprocess data before training**

---

## 🚀 Step-by-Step Workflow

### Step 1: Preprocess Data for Case 1 (Question Only)

```bash
python feedbackqa/preprocess_ppo_case1_question_only.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --test_file feedbackqa/feedback_test_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case1_question_only \
    --seed 42
```

**Expected output:**
```
Processing 5400 examples for Case 1 (Question Only)...
Saved 5400 examples to ~/data/feedback_qa_ppo/case1_question_only/train.parquet
...
```

### Step 2: Preprocess Data for Case 2 (With Feedback)

```bash
python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --test_file feedbackqa/feedback_test_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case2_with_feedback \
    --seed 42
```

**Expected output:**
```
Processing 5400 examples for Case 2 (With Feedback)...
Examples WITH context: 5400 (100.0%)
Saved 5400 examples to ~/data/feedback_qa_ppo/case2_with_feedback/train.parquet
...
```

### Step 3: Verify Preprocessed Data

```bash
# Check Case 1
ls -lh ~/data/feedback_qa_ppo/case1_question_only/
# Should show: train.parquet, valid.parquet, test.parquet

# Check Case 2
ls -lh ~/data/feedback_qa_ppo/case2_with_feedback/
# Should show: train.parquet, valid.parquet, test.parquet
```

### Step 4: Train Case 1 (Baseline)

```bash
bash feedbackqa/run_ppo_case1_question_only.sh
```

**What happens:**
- Downloads FsfairX-LLaMA3-RM-v0.1 (pre-trained reward model)
- Checks data files exist
- Runs PPO training with question-only prompts
- Saves model checkpoints
- Logs to W&B

**Expected time:** ~30-60 minutes per epoch (8xA100)

### Step 5: Train Case 2 (Experimental)

```bash
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

**What happens:**
- Uses same reward model
- Runs PPO training with feedback-augmented prompts
- Saves model checkpoints
- Logs to W&B

**Expected time:** ~90-150 minutes per epoch (longer sequences)

### Step 6: Compare Results in W&B

Navigate to your W&B dashboard:
- Project: `feedback_qa_experiment`
- Runs:
  - `case1_question_only_baseline`
  - `case2_with_feedback_experimental`

Compare metrics:
- Reward scores
- Training loss
- Validation metrics

---

## ⚠️ Important Notes

### Using Pre-trained Reward Model

Both scripts are configured to use `FsfairX-LLaMA3-RM-v0.1`:
```bash
reward_model.model.path="$HOME/models/FsfairX-LLaMA3-RM-v0.1"
```

**Pros:**
- ✅ Pre-trained and tested
- ✅ Already has `num_labels=1` (verl-compatible)
- ✅ No need to train your own RM

**Cons:**
- ⚠️ May not be specifically tuned for your FeedbackQA domain
- ⚠️ Chat template might differ from your policy model

### If You Want to Train Your Own RM

If you want domain-specific rewards:

1. Train with fixed `rm_train.py`:
```bash
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./feedback_qa_reward_model \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

2. Verify:
```bash
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```

3. Update training scripts to use your model:
```bash
# Change this line in both scripts:
reward_model.model.path="./feedback_qa_reward_model/final_model"
```

---

## 🐛 Troubleshooting

### Error: `AssertionError: val_metrics={}`

**Cause:** Validation data doesn't exist  
**Solution:** Run preprocessing scripts (Steps 1-2)

### Error: `size mismatch for weight: torch.Size([2, 4096]) vs torch.Size([1, 4096])`

**Cause:** Using reward model with wrong architecture  
**Solution:** Use pre-trained model or retrain with fixed `rm_train.py`

### Error: `Data directory not found`

**Cause:** Haven't run preprocessing  
**Solution:** Run Steps 1-2 above

### OOM (Out of Memory)

**For Case 1:**
```bash
# Reduce batch size
data.train_batch_size=128  # instead of 256
```

**For Case 2:**
```bash
# Already reduced, but can go lower
data.train_batch_size=64   # instead of 128
data.max_prompt_length=800 # instead of 1024
```

---

## 📊 Expected Results

### Training Progress

**Case 1 (Baseline):**
- Epoch 1: Initial rewards ~0.3-0.5
- Epoch 5: Improving ~0.6-0.7
- Epoch 10: Should plateau ~0.7-0.8

**Case 2 (Feedback-Augmented):**
- Epoch 1: Initial rewards ~0.4-0.6 (slightly higher)
- Epoch 5: Improving ~0.7-0.8
- Epoch 10: Should plateau ~0.8-0.9 (hopefully higher than Case 1!)

### Success Criteria

✅ **Hypothesis confirmed** if:
- Case 2 final reward > Case 2 final reward
- Case 2 converges faster
- Case 2 validation metrics better

❌ **Hypothesis rejected** if:
- No significant difference
- Case 1 performs better
- Both converge to similar values

---

## 📁 File Checklist

Before training, ensure these exist:

### Data Files
- [ ] `feedbackqa/feedback_train_ppo.json`
- [ ] `feedbackqa/feedback_valid_ppo.json`
- [ ] `feedbackqa/feedback_test_ppo.json`

### Preprocessed Data (after Steps 1-2)
- [ ] `~/data/feedback_qa_ppo/case1_question_only/train.parquet`
- [ ] `~/data/feedback_qa_ppo/case1_question_only/valid.parquet`
- [ ] `~/data/feedback_qa_ppo/case2_with_feedback/train.parquet`
- [ ] `~/data/feedback_qa_ppo/case2_with_feedback/valid.parquet`

### Scripts
- [ ] `feedbackqa/preprocess_ppo_case1_question_only.py`
- [ ] `feedbackqa/preprocess_ppo_case2_with_feedback.py`
- [ ] `feedbackqa/run_ppo_case1_question_only.sh`
- [ ] `feedbackqa/run_ppo_case2_with_feedback.sh`

### Models (auto-downloaded)
- [ ] `~/models/FsfairX-LLaMA3-RM-v0.1/` (downloaded by training scripts)

---

## 🎯 Quick Start (TL;DR)

```bash
# 1. Preprocess both cases
python feedbackqa/preprocess_ppo_case1_question_only.py
python feedbackqa/preprocess_ppo_case2_with_feedback.py

# 2. Train both cases
bash feedbackqa/run_ppo_case1_question_only.sh
bash feedbackqa/run_ppo_case2_with_feedback.sh

# 3. Compare in W&B
# Navigate to: https://wandb.ai/your-username/feedback_qa_experiment
```

---

**Current Status:** ✅ Ready to preprocess and train!

