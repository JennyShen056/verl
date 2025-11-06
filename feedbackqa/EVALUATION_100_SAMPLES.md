# Evaluation on 100 Test Samples

## 🎯 **Overview**

For faster evaluation, the pipeline now:
1. Randomly selects **100 samples** from `feedback_test_ppo.json` (with seed=42 for reproducibility)
2. Uses the **SAME 100 samples** for both Case 1 and Case 2
3. Ensures fair comparison between cases

---

## 🚀 **Quick Start**

```bash
# Run complete evaluation on 100 samples
bash feedbackqa/run_full_evaluation.sh 1
```

**Time:** ~15-30 minutes (vs 1-2 hours for full test set)

---

## 📋 **What Happens**

### **Step 0: Create Test Subset**
```bash
python feedbackqa/create_test_subset.py \
    --input_file feedbackqa/feedback_test_ppo.json \
    --output_file feedbackqa/feedback_test_subset_100.json \
    --num_samples 100 \
    --seed 42
```

**Output:** `feedbackqa/feedback_test_subset_100.json` (100 samples)

**Key:** Uses `seed=42` so the same 100 samples are selected every time!

### **Steps 1-7: Full Pipeline**
- Merge checkpoints (Case 1 & 2)
- Run inference (using subset for both cases)
- Evaluate with reward model (both cases)
- Compare statistically

---

## ✅ **Key Features**

### **1. Reproducibility**
- Fixed seed (42) ensures same samples every time
- Both cases use identical test set
- Fair comparison guaranteed

### **2. Speed**
- **100 samples**: ~15-30 min
- **Full test set** (~1000+ samples): ~1-2 hours
- **10x faster** for quick iterations!

### **3. Same Samples for Both Cases**
```
feedback_test_subset_100.json  ← Created once at Step 0
         ↓                 ↓
    Case 1 Inference   Case 2 Inference  ← Both use same file
         ↓                 ↓
    Case 1 Eval       Case 2 Eval
         └─────────┬─────────┘
              Comparison  ← Fair comparison on identical samples
```

---

## 📁 **Output Files**

```
feedbackqa/feedback_test_subset_100.json  ← 100 test samples (same for both)

outputs/
├── case1_predictions.json                ← 100 predictions (Case 1)
├── case1_evaluation.json                 ← 100 scores (Case 1)
├── case2_predictions.json                ← 100 predictions (Case 2)
├── case2_evaluation.json                 ← 100 scores (Case 2)
└── comparison_report.json                ← Statistical comparison
```

---

## 🔍 **Manual Subset Creation**

If you want to create different subsets:

```bash
# Create subset with different size
python feedbackqa/create_test_subset.py \
    --input_file feedbackqa/feedback_test_ppo.json \
    --output_file feedbackqa/feedback_test_subset_50.json \
    --num_samples 50 \
    --seed 42

# Create subset with different seed (different samples)
python feedbackqa/create_test_subset.py \
    --input_file feedbackqa/feedback_test_ppo.json \
    --output_file feedbackqa/feedback_test_subset_100_v2.json \
    --num_samples 100 \
    --seed 123
```

---

## 📊 **Statistical Validity**

### **Is 100 samples enough?**

**Yes, for most cases:**
- Statistical tests work with n=100
- Cohen's d effect size is valid
- p-values are interpretable

**Power analysis:**
```
n=100, α=0.05, medium effect (d=0.5)
→ Power ≈ 85% (good!)

n=100, α=0.05, small effect (d=0.3)
→ Power ≈ 50% (may miss small effects)
```

**Recommendation:**
- **Initial testing**: 100 samples (fast iteration)
- **Final evaluation**: Full test set (highest confidence)

---

## 🔄 **Workflow**

### **During Development (100 samples)**
```bash
# Quick evaluation for testing
bash feedbackqa/run_full_evaluation.sh 1
# Check if code works, iterate quickly
```

### **Final Evaluation (Full test set)**

To use full test set, edit `run_full_evaluation.sh`:

```bash
# Comment out Step 0
# python feedbackqa/create_test_subset.py ...

# Change TEST_FILE
# TEST_FILE="feedbackqa/feedback_test_subset_100.json"
TEST_FILE="feedbackqa/feedback_test_ppo.json"  # Use full test set
```

Or manually:

```bash
# Manual evaluation on full set
python feedbackqa/inference.py \
    --model_path <model_path> \
    --test_file feedbackqa/feedback_test_ppo.json \
    --output_file outputs/case1_predictions_full.json
```

---

## ⚠️ **Important Notes**

### **1. Same Samples Guarantee**
✅ Both cases use the **exact same** 100 samples because:
- Subset created **once** at Step 0
- Both cases read from **same file** (`feedback_test_subset_100.json`)
- Fixed seed (42) ensures reproducibility

### **2. Reproducibility**
Running the script multiple times will:
- ✅ Select the **same** 100 samples (seed=42)
- ✅ Evaluate on **same** test set
- ✅ Produce **comparable** results across runs

### **3. Different Subsets for Different Experiments**
If you want to test on different samples:
```bash
# Change seed in run_full_evaluation.sh
--seed 42  → --seed 123
```

---

## 🎓 **Tips**

### **Quick Testing**
```bash
# Test on even smaller subset for debugging
python feedbackqa/create_test_subset.py \
    --input_file feedbackqa/feedback_test_ppo.json \
    --output_file feedbackqa/feedback_test_subset_10.json \
    --num_samples 10 \
    --seed 42

# Then edit run_full_evaluation.sh to use 10 samples
```

### **Multiple Random Splits**
```bash
# Run multiple evaluations with different samples
for seed in 42 123 456; do
    python feedbackqa/create_test_subset.py \
        --input_file feedbackqa/feedback_test_ppo.json \
        --output_file feedbackqa/feedback_test_subset_100_seed${seed}.json \
        --num_samples 100 \
        --seed $seed
    
    # Run evaluation...
done

# Average results across splits for robustness
```

---

## 🔢 **Expected Results**

### **With 100 Samples**

```
📊 Overall Metrics:
Metric               Case 1       Case 2   Difference   % Change
--------------------------------------------------------------------------------
mean_reward          0.6234       0.7456       0.1222      19.6% ✓

📈 Statistical Tests:
T-test:
  p-value: 0.0089 ✓ Significant (α=0.05)

Effect Size:
  Cohen's d: 0.6234
  Interpretation: MEDIUM

Pairwise Comparison:
  Case 2 wins: 68 (68.0%)
  Ties: 5
  Case 1 wins: 27 (27.0%)
```

**Interpretation:**
- With 100 samples, you can detect medium-large effects
- Confidence intervals will be wider than full test set
- Results are still statistically valid!

---

## ✅ **Summary**

| Aspect | 100 Samples | Full Test Set |
|--------|-------------|---------------|
| **Time** | ~15-30 min | ~1-2 hours |
| **Statistical Power** | Good for medium+ effects | Best |
| **Use Case** | Quick testing, iteration | Final evaluation |
| **Fair Comparison** | ✅ Same samples both cases | ✅ Same samples both cases |
| **Reproducible** | ✅ Fixed seed | ✅ Fixed order |

---

**Status:** 🟢 **Ready for quick evaluation!**

Run: `bash feedbackqa/run_full_evaluation.sh 1`

Both cases will be evaluated on the **same 100 random samples** for a fair, fast comparison! 🚀

