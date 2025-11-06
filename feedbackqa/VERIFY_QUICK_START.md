# Quick Start: Verify BCE Loss Training

## 🚀 Fastest Way to Verify

```bash
python feedbackqa/verify_bce_training.py
```

That's it! This comprehensive script checks everything.

---

## ✅ What You'll See

### If Everything is Correct:

```
================================================================================
BCE LOSS VERIFICATION
================================================================================

1. Loading model...
   ✓ Model loaded successfully
   Model type: LlamaForTokenClassification
   num_labels: 1  ✅

2. Checking model configuration...
   ✓ num_labels=1 (correct for verl)

3. Testing model forward pass...
   ✓ Model predictions working

4. Summary
   Correct predictions: 2/2  ✅

5. Verifying output shapes...
   ✓ Shape correct: [batch_size, seq_len, 1]

6. Checking training summary...
   loss_function: binary_cross_entropy  ✅
   verl_compatible: True  ✅
   ✓ Training used BCE loss

================================================================================
VERIFICATION COMPLETE
================================================================================

✅ All checks passed! Model is working correctly with BCE loss.

Model is verl-compatible and ready for PPO training!
```

---

## 📋 3-Step Verification

### Before Training

Check the code has BCE:
```bash
grep "binary_cross_entropy" feedbackqa/rm_train.py
```

Should show: `F.binary_cross_entropy_with_logits` ✅

### During Training

Watch for in the logs:
```
Loss function: Binary Cross-Entropy (BCE) - optimal for binary labels
```

### After Training

Run verification:
```bash
python feedbackqa/verify_bce_training.py
```

---

## 🔍 Alternative: Manual Checks

```bash
# Check config file
cat ./feedback_qa_reward_model/final_model/config.json | grep num_labels
# Should show: "num_labels": 1  ✅

# Check training summary
cat ./feedback_qa_reward_model/training_summary.json | grep loss_function
# Should show: "loss_function": "binary_cross_entropy"  ✅

# Run verl verification
python feedbackqa/verify_trained_model.py \
    --model_path ./feedback_qa_reward_model/final_model
# Should show: ✓ VERL COMPATIBILITY VERIFIED ✅
```

---

## ❓ What If Model Doesn't Exist Yet?

Run without model test:
```bash
python feedbackqa/verify_bce_training.py --skip_model_test
```

This will:
- ✓ Verify BCE computation math
- ✓ Show gradient comparisons
- ✓ Confirm implementation is correct

Then train your model and run full verification!

---

## ✅ Success = Ready for PPO!

When verification passes:

1. ✅ Model uses BCE loss
2. ✅ Model is verl-compatible  
3. ✅ Ready to preprocess PPO data
4. ✅ Ready to run PPO training

Next steps:
```bash
python feedbackqa/preprocess_ppo_case1_question_only.py
python feedbackqa/preprocess_ppo_case2_with_feedback.py
bash feedbackqa/run_ppo_case1_question_only.sh
```

---

## 📚 More Details

- Full guide: `VERIFICATION_GUIDE.md`
- BCE explanation: `BCE_VS_MSE_LOSS.md`
- Visual comparison: `LOSS_FUNCTION_COMPARISON.md`

---

**TL;DR**: Run `python feedbackqa/verify_bce_training.py` and look for ✅

