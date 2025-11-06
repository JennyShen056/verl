# Quick Command Reference 🚀

## **After Training Completes**

### **Option 1: Automated (Recommended)**
```bash
# Run complete evaluation pipeline
bash feedbackqa/run_full_evaluation.sh 1

# Results saved to outputs/comparison_report.json
```

### **Option 2: Manual Control**

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

# Step 3: Evaluate
python feedbackqa/evaluate.py \
    --predictions_file outputs/case1_predictions.json \
    --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --output_file outputs/case1_evaluation.json

python feedbackqa/evaluate.py \
    --predictions_file outputs/case2_predictions.json \
    --reward_model_path sfairXC/FsfairX-LLaMA3-RM-v0.1 \
    --output_file outputs/case2_evaluation.json

# Step 4: Compare
python feedbackqa/compare_results.py \
    --case1_file outputs/case1_evaluation.json \
    --case2_file outputs/case2_evaluation.json \
    --output_file outputs/comparison_report.json
```

## **Check Results**

```bash
# View comparison
cat outputs/comparison_report.json | jq '.statistical_tests'

# View metrics
cat outputs/case1_evaluation.json | jq '.metrics'
cat outputs/case2_evaluation.json | jq '.metrics'
```

## **List Checkpoints**

```bash
ls -d checkpoints/feedback_qa_experiment/*/global_step_*
```

## **Evaluate Different Checkpoint**

```bash
# Evaluate global_step_5
bash feedbackqa/run_full_evaluation.sh 5
```

---

**Files Created:**
- ✅ `merge_checkpoint.sh` - Checkpoint conversion
- ✅ `inference.py` - Generate predictions
- ✅ `evaluate.py` - Score with RM
- ✅ `compare_results.py` - Statistical comparison
- ✅ `run_full_evaluation.sh` - Automated pipeline

**Documentation:**
- 📖 `INFERENCE_AND_EVALUATION_GUIDE.md` - Detailed guide
- 📖 `COMPLETE_PIPELINE_SUMMARY.md` - Full overview
- 📖 `THIS_FILE.md` - Quick reference

