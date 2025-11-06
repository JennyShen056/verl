#!/usr/bin/env python3
"""
Complete Experimental Pipeline: Question-Only vs Feedback-Augmented Learning
=============================================================================

This guide walks you through the complete experimental pipeline to test:
"Does showing models examples with feedback improve answer quality?"

Total Time: ~24-30 hours (can run in parallel)
"""

# ============================================================================
# EXPERIMENTAL SETUP
# ============================================================================

"""
Research Question:
    Does feedback-augmented training (showing Q+A+Feedback examples) 
    improve model performance compared to question-only training?

Experimental Design:
    - Control (Case 1): Model sees ONLY questions
    - Treatment (Case 2): Model sees questions + example Q+A+Feedback
    - Both scored by same trained reward model
    - Fair comparison at test time (both get just questions)

Expected Outcome:
    If Case 2 > Case 1: Feedback examples help learning
    If Case 1 >= Case 2: Direct RL is sufficient
"""

# ============================================================================
# STEP 1: PREPARE ENVIRONMENT
# ============================================================================

"""
1.1 Install Dependencies
"""
# Ensure you have verl installed
# pip install -r requirements.txt

"""
1.2 Verify Data Files
"""
# You should have these files:
# - feedbackqa/feedback_train_rm.json     (for training RM)
# - feedbackqa/feedback_valid_rm.json     (for validating RM)
# - feedbackqa/feedback_test_rm.json      (for testing RM)
# - feedbackqa/feedback_train_ppo.json    (for PPO training)
# - feedbackqa/feedback_valid_ppo.json    (for PPO validation)
# - feedbackqa/feedback_test_ppo.json     (for PPO testing)

"""
1.3 Check GPU Availability
"""
# nvidia-smi
# Recommended: 8x GPUs for efficient training
# Minimum: 4x GPUs (adjust batch sizes accordingly)

# ============================================================================
# STEP 2: TRAIN REWARD MODEL
# ============================================================================

"""
2.1 Train the Reward Model (verl-compatible)
"""
# Time: ~2-3 hours
print("="*60)
print("STEP 2.1: Training Reward Model")
print("="*60)

# Command:
"""
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./feedback_qa_reward_model \
    --batch_size 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --max_length 1024 \
    --seed 42
"""

"""
2.2 Verify Reward Model
"""
print("\nSTEP 2.2: Verifying Reward Model Compatibility")

# Command:
"""
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
"""

# Expected output:
"""
✓ num_labels=1 (correct for verl)
✓ Uses TokenClassification (correct for verl)
✓ Model loaded successfully
✓ Output shape is correct
✅ SUCCESS! Your model is verl-compatible!
"""

# ============================================================================
# STEP 3: PREPROCESS DATA FOR BOTH CASES
# ============================================================================

"""
3.1 Preprocess Case 1: Question Only (Baseline)
"""
# Time: ~5 minutes
print("\n" + "="*60)
print("STEP 3.1: Preprocessing Case 1 (Question Only)")
print("="*60)

# Command:
"""
python feedbackqa/preprocess_ppo_case1_question_only.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --test_file feedbackqa/feedback_test_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case1_question_only \
    --seed 42
"""

# Output location:
"""
~/data/feedback_qa_ppo/case1_question_only/
    ├── train.parquet
    ├── valid.parquet
    └── test.parquet
"""

"""
3.2 Preprocess Case 2: With Feedback (Experimental)
"""
# Time: ~5 minutes
print("\nSTEP 3.2: Preprocessing Case 2 (With Feedback)")

# Command:
"""
python feedbackqa/preprocess_ppo_case2_with_feedback.py \
    --train_file feedbackqa/feedback_train_ppo.json \
    --valid_file feedbackqa/feedback_valid_ppo.json \
    --test_file feedbackqa/feedback_test_ppo.json \
    --local_save_dir ~/data/feedback_qa_ppo/case2_with_feedback \
    --seed 42
"""

# Output location:
"""
~/data/feedback_qa_ppo/case2_with_feedback/
    ├── train.parquet
    ├── valid.parquet
    └── test.parquet
"""

"""
3.3 Verify Preprocessed Data
"""
print("\nSTEP 3.3: Verifying Preprocessed Data")

# Python verification:
"""
import pandas as pd

# Check Case 1
df1 = pd.read_parquet("~/data/feedback_qa_ppo/case1_question_only/train.parquet")
print(f"Case 1 train size: {len(df1)}")
print(f"Case 1 prompt example: {df1['prompt'].iloc[0]}")

# Check Case 2
df2 = pd.read_parquet("~/data/feedback_qa_ppo/case2_with_feedback/train.parquet")
print(f"Case 2 train size: {len(df2)}")
print(f"Case 2 prompt example (shows feedback): {df2['prompt'].iloc[0]}")
"""

# ============================================================================
# STEP 4: TRAIN BOTH MODELS WITH PPO
# ============================================================================

"""
4.1 Train Case 1: Question Only (Baseline)
"""
# Time: ~10-12 hours
print("\n" + "="*60)
print("STEP 4.1: Training Case 1 (Baseline)")
print("="*60)

# Command:
"""
bash feedbackqa/run_ppo_case1_question_only.sh
"""

# This will:
"""
1. Load preprocessed Case 1 data (questions only)
2. Initialize policy model (Llama-3.2-3B-Instruct)
3. Load trained reward model
4. Run PPO training for 10 epochs
5. Save checkpoints every 5 epochs
6. Log to W&B: project=feedback_qa_experiment, name=case1_question_only_baseline
"""

"""
4.2 Train Case 2: With Feedback (Experimental)
"""
# Time: ~10-12 hours
# NOTE: Can run in parallel with Case 1 if you have enough GPUs

print("\nSTEP 4.2: Training Case 2 (Experimental)")

# Command:
"""
bash feedbackqa/run_ppo_case2_with_feedback.sh
"""

# This will:
"""
1. Load preprocessed Case 2 data (with feedback examples)
2. Initialize policy model (Llama-3.2-3B-Instruct)
3. Load trained reward model
4. Run PPO training for 10 epochs
5. Save checkpoints every 5 epochs
6. Log to W&B: project=feedback_qa_experiment, name=case2_with_feedback_experimental
"""

"""
4.3 Monitor Training
"""
print("\nSTEP 4.3: Monitoring Training Progress")

# Check W&B dashboard:
"""
Project: feedback_qa_experiment
Runs:
  - case1_question_only_baseline
  - case2_with_feedback_experimental

Key metrics to watch:
  - actor/reward (higher is better)
  - val-core/<dataset>/reward/mean@1 (validation performance)
  - training/loss (should decrease)
  - training/entropy (exploration)
"""

# Check logs:
"""
tail -f logs/case1_question_only_baseline.log
tail -f logs/case2_with_feedback_experimental.log
"""

# ============================================================================
# STEP 5: EVALUATE AND COMPARE
# ============================================================================

"""
5.1 Load Best Checkpoints
"""
print("\n" + "="*60)
print("STEP 5: Evaluation and Comparison")
print("="*60)

# Checkpoints saved to:
"""
./outputs/case1_question_only_baseline/global_step_<N>/actor/
./outputs/case2_with_feedback_experimental/global_step_<N>/actor/
"""

"""
5.2 Compare Training Metrics
"""
# In W&B, compare:
"""
1. Final reward scores (higher = better)
2. Convergence speed (epochs to peak performance)
3. Stability (variance in rewards)
4. Validation performance
"""

"""
5.3 Inference Test
"""
# Test both models on same questions:
"""
python feedbackqa/test_inference.py \
    --model1_path ./outputs/case1_question_only_baseline/global_step_<N>/actor/ \
    --model2_path ./outputs/case2_with_feedback_experimental/global_step_<N>/actor/ \
    --test_file feedbackqa/feedback_test_ppo.json \
    --reward_model_path ./feedback_qa_reward_model/final_model \
    --output_file comparison_results.json
"""

"""
5.4 Generate Comparison Report
"""
print("\nSTEP 5.4: Generating Comparison Report")

# Analysis script (create this):
"""
python feedbackqa/analyze_results.py \
    --wandb_project feedback_qa_experiment \
    --run1_name case1_question_only_baseline \
    --run2_name case2_with_feedback_experimental \
    --output comparison_report.md
"""

# ============================================================================
# STEP 6: INTERPRET RESULTS
# ============================================================================

"""
6.1 Statistical Comparison
"""
print("\n" + "="*60)
print("STEP 6: Results Interpretation")
print("="*60)

# Key questions:
"""
1. Which model achieved higher final reward scores?
2. Which model converged faster?
3. Which model is more stable during training?
4. How do human evaluations compare?
"""

"""
6.2 Example Analysis
"""
# Look at example generations:
"""
Question: "How do I get help finding a job?"

Case 1 (Question Only) Generated:
    "Visit job websites and search for positions..."
    RM Score: 0.65

Case 2 (With Feedback) Generated:
    "In this job market, visit the Jobs Hub for vacancies. 
     Consider sectors like health, logistics, retail..."
    RM Score: 0.89

Analysis: Case 2 learned to include specific details and 
          sector recommendations from feedback examples.
"""

"""
6.3 Conclusions
"""
# Interpret based on results:
"""
Scenario A: Case 2 significantly better
    → Feedback-augmented learning is effective
    → Models benefit from seeing quality examples
    → Recommendation: Use feedback examples in training

Scenario B: No significant difference
    → Direct RL (Case 1) is sufficient
    → Feedback examples don't add value
    → Recommendation: Stick with simpler approach

Scenario C: Case 1 better
    → Feedback examples may confuse or distract
    → Direct RL more efficient
    → Recommendation: Question-only training preferred
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Common Issues and Solutions:
"""

"""
Issue 1: OOM during training
Solution: 
    - Reduce batch sizes
    - Enable offloading: param_offload=True
    - Reduce max_prompt_length
    - Use gradient checkpointing
"""

"""
Issue 2: RM not loading
Solution:
    - Verify model with verify_trained_model.py
    - Check num_labels=1
    - Ensure AutoModelForTokenClassification
"""

"""
Issue 3: Poor training performance
Solution:
    - Check RM quality (accuracy on test set)
    - Verify data preprocessing
    - Adjust learning rates
    - Increase training epochs
"""

"""
Issue 4: Models diverge during training
Solution:
    - Lower learning rate
    - Increase KL penalty
    - Use gradient clipping
    - Check for data quality issues
"""

# ============================================================================
# EXPECTED TIMELINE
# ============================================================================

"""
Complete Experiment Timeline:

Day 1:
    - Setup environment (1 hour)
    - Train reward model (3 hours)
    - Preprocess data (10 minutes)
    - Start Case 1 training (overnight)

Day 2:
    - Monitor Case 1 training (12 hours)
    - Start Case 2 training (overnight)

Day 3:
    - Monitor Case 2 training (12 hours)
    - Run evaluation (2 hours)
    - Generate reports (1 hour)
    - Analyze results (2 hours)

Total: ~3 days (with serial training)
Total: ~2 days (with parallel training on multiple GPU clusters)
"""

# ============================================================================
# SUCCESS CHECKLIST
# ============================================================================

"""
Before claiming completion:

[✓] Reward model trained and verified
[✓] Both datasets preprocessed correctly
[✓] Case 1 training completed successfully
[✓] Case 2 training completed successfully
[✓] Both models evaluated on same test set
[✓] Results logged to W&B
[✓] Statistical comparison performed
[✓] Example generations inspected
[✓] Conclusions documented
[✓] Results reproducible
"""

# ============================================================================
# NEXT STEPS
# ============================================================================

"""
After completing the experiment:

1. Write up results
2. Share findings with team
3. If successful, apply to other domains
4. Consider ablations:
   - Different amounts of feedback
   - Different feedback formats
   - Different base models
   - Different RM architectures
"""

print("\n" + "="*60)
print("Experiment Setup Complete!")
print("="*60)
print("Follow the steps above to run the complete experiment.")
print("Good luck! 🚀")

