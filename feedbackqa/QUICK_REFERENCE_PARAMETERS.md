# Quick Reference: Training Parameters

**Based on**: Token length analysis of `feedback_train_ppo.json` (5,400 examples)  
**Tokenizer**: `meta-llama/Llama-3.1-8B-Instruct`

---

## 📊 At a Glance

| Parameter | Case 1<br>(Question Only) | Case 2<br>(With Feedback) | Ratio |
|-----------|---------------------------|---------------------------|-------|
| **max_prompt_length** | 512 | 1024 | 2.0x |
| **max_response_length** | 565 | 565 | 1.0x |
| **train_batch_size** | 256 | 128 | 0.5x |
| **ppo_mini_batch_size** | 128 | 64 | 0.5x |
| **Mean tokens/example** | 212 | 485 | 2.3x |
| **Training speed** | 1.0x | 0.43x | 2.3x slower |

---

## 🎯 Copy-Paste Ready Configs

### Case 1: Question Only

```bash
# Data parameters
data.max_prompt_length=512
data.max_response_length=565
data.train_batch_size=256

# Actor parameters
actor_rollout_ref.actor.ppo_mini_batch_size=128
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8

# Critic parameters
critic.ppo_micro_batch_size_per_gpu=16
```

### Case 2: With Previous Answer + Feedback

```bash
# Data parameters
data.max_prompt_length=1024
data.max_response_length=565
data.train_batch_size=128

# Actor parameters  
actor_rollout_ref.actor.ppo_mini_batch_size=64
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4

# Critic parameters
critic.ppo_micro_batch_size_per_gpu=8
```

---

## 📈 Actual Data Statistics

### Raw Token Lengths

| Component | Min | Mean | Median | 95th % | Max |
|-----------|-----|------|--------|--------|-----|
| **Question** | 5 | 14.4 | 14 | 23 | 36 |
| **Answer** | 2 | 196.3 | 144 | 514 | 1,957 |
| **Feedback** | 6 | 25.0 | 22 | 47 | 101 |

### Case 1 Prompts (Question Only)

| Metric | Value |
|--------|-------|
| Mean | 15.4 tokens |
| Median | 15 tokens |
| 95th percentile | 24 tokens |
| Max | 37 tokens |

**Coverage**: `max_prompt_length=512` covers **100%** of examples ✅

### Case 2 Prompts (With Feedback Context)

| Metric | Value |
|--------|-------|
| Mean | 288.6 tokens |
| Median | 232 tokens |
| 95th percentile | 635 tokens |
| Max | 1,613 tokens |

**Coverage**: `max_prompt_length=1024` covers **~95%** of examples ✅

---

## 🚀 Training Speed Estimates

**Hardware**: 8 x A100 (80GB)  
**Dataset**: 5,400 examples  
**Epochs**: 10

### Case 1
- Global batch size: 256 × 8 = **2,048**
- Steps per epoch: **~2.6**
- Total steps: **~26**
- Time: **30-60 minutes**

### Case 2
- Global batch size: 128 × 8 = **1,024**
- Steps per epoch: **~5.3**
- Total steps: **~53**
- Time: **90-150 minutes**

---

## ⚠️ Troubleshooting: OOM Errors

### If Case 2 runs out of memory:

**Step 1**: Reduce batch size
```bash
data.train_batch_size=64  # halve it
actor_rollout_ref.actor.ppo_mini_batch_size=32
```

**Step 2**: Reduce max lengths
```bash
data.max_prompt_length=800  # covers ~90%
data.max_response_length=450  # covers ~85%
```

**Step 3**: Enable offloading
```bash
actor_rollout_ref.actor.fsdp_config.param_offload=True
```

**Step 4**: Reduce micro batch sizes
```bash
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2
critic.ppo_micro_batch_size_per_gpu=4
```

---

## ✅ Pre-Flight Checklist

Before starting training:

- [ ] Reward model trained (`feedback_qa_reward_model/final_model/` exists)
- [ ] Data preprocessed for Case 1 (`~/data/feedback_qa_ppo/case1_question_only/train.parquet`)
- [ ] Data preprocessed for Case 2 (`~/data/feedback_qa_ppo/case2_with_feedback/train.parquet`)
- [ ] Training scripts updated with correct parameters
- [ ] W&B project configured (`feedback_qa_experiment`)
- [ ] GPU resources available (8xA100 recommended)

---

## 🔍 Validation Commands

**Check data**:
```bash
python feedbackqa/inspect_preprocessed_data.py \
    --case1_file ~/data/feedback_qa_ppo/case1_question_only/train.parquet \
    --case2_file ~/data/feedback_qa_ppo/case2_with_feedback/train.parquet
```

**Run analysis**:
```bash
python feedbackqa/analyze_prompt_response_lengths.py
```

**Verify RM**:
```bash
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
```

---

## 📝 Key Files

| Purpose | File |
|---------|------|
| **Analysis script** | `analyze_prompt_response_lengths.py` |
| **Analysis results** | `ANALYSIS_RESULTS.md` |
| **Case 1 preprocessing** | `preprocess_ppo_case1_question_only.py` |
| **Case 2 preprocessing** | `preprocess_ppo_case2_with_feedback.py` |
| **Case 1 training** | `run_ppo_case1_question_only.sh` |
| **Case 2 training** | `run_ppo_case2_with_feedback.sh` |
| **Experiment design** | `EXPERIMENTAL_DESIGN.md` |
| **Full guide** | `RUN_EXPERIMENT.md` |

---

## 🎯 Expected Outcome

If feedback helps models learn:
- Case 2 reward scores > Case 1 reward scores
- Case 2 validation loss < Case 1 validation loss
- Case 2 generates higher quality answers when evaluated

Compare in Weights & Biases:
- Project: `feedback_qa_experiment`
- Runs: `case1_question_only_baseline` vs `case2_with_feedback_experimental`

---

**Last Updated**: After token length analysis  
**Status**: ✅ Ready for training

