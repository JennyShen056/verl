# HelpSteer3 PPO Training Pipeline

Complete pipeline for training and evaluating PPO models on the HelpSteer3 feedback dataset, comparing two cases:
- **Case 1**: Question Only (Baseline) - Model receives only conversation context
- **Case 2**: With Feedback - Model receives conversation + example response + feedback

## Dataset

- **Source**: `nvidia/HelpSteer3` (feedback subset)
- **Splits**: 
  - Train: ~X samples (from HuggingFace train split)
  - Validation: ~1.5k samples (from HF validation, excluding test)
  - Test: 500 samples (split from HF validation, same for both cases)

## Quick Start

### Step 1: Preprocess Data

```bash
# Case 1: Question Only
python helpsteer3/preprocess_ppo_helpsteer3_case1_question_only.py \
    --splits train validation test \
    --local_save_dir ~/data/helpsteer3_ppo/case1_question_only \
    --seed 42 \
    --test_size 500

# Case 2: With Feedback
python helpsteer3/preprocess_ppo_helpsteer3_case2_with_feedback.py \
    --splits train validation test \
    --local_save_dir ~/data/helpsteer3_ppo/case2_with_feedback \
    --seed 42 \
    --test_size 500
```

**Important**: Use the same `--seed` and `--test_size` for both cases to ensure identical test splits!

### Step 2: Train Models

```bash
# Case 1: Question Only (Baseline)
bash helpsteer3/run_ppo_case1_question_only.sh

# Case 2: With Feedback (Experimental)
bash helpsteer3/run_ppo_case2_with_feedback.sh
```

Monitor training in Weights & Biases:
- Project: `helpsteer3_ppo_experiment`
- Experiments: `helpsteer3_case1_question_only_baseline` vs `helpsteer3_case2_with_feedback_experimental`

### Step 3: Evaluate and Compare

```bash
# Edit the script to set the checkpoint steps you want to evaluate
# Then run the full evaluation pipeline
bash helpsteer3/run_full_evaluation.sh
```

This will:
1. Merge checkpoints to HuggingFace format
2. Run inference on test set (500 samples)
3. Evaluate using reward model
4. Compare results statistically

## File Structure

```
helpsteer3/
├── preprocess_ppo_helpsteer3_case1_question_only.py  # Data preprocessing for Case 1
├── preprocess_ppo_helpsteer3_case2_with_feedback.py  # Data preprocessing for Case 2
├── run_ppo_case1_question_only.sh                     # Training script for Case 1
├── run_ppo_case2_with_feedback.sh                     # Training script for Case 2
├── inference.py                                        # Generate predictions on test set
├── evaluate.py                                         # Evaluate predictions with RM
├── compare_results.py                                  # Statistical comparison
├── run_full_evaluation.sh                              # Complete evaluation pipeline
└── README.md                                           # This file
```

## Reward Model

The evaluation uses **sfairXC/FsfairX-LLaMA3-RM-v0.1** (hosted as `Jennny/llama3_help_rm`).

This is a LLaMA-3 based reward model that scores conversation quality. It uses a pipeline-based approach:

```python
from transformers import AutoTokenizer, pipeline

rm_tokenizer = AutoTokenizer.from_pretrained("sfairXC/FsfairX-LLaMA3-RM-v0.1")
rm_pipe = pipeline(
    "sentiment-analysis",
    model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
    device=0,
    tokenizer=rm_tokenizer,
    model_kwargs={"torch_dtype": torch.bfloat16}
)

# Evaluate a conversation
chat = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
]
text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
text = text.replace(tokenizer.bos_token, "")  # Remove BOS token

pipe_outputs = rm_pipe([text], return_all_scores=True, function_to_apply="none")
reward = pipe_outputs[0][0]["score"]
```

## Key Configuration Differences

### Case 1 (Baseline)
- **Max prompt length**: 1024 tokens
- **Batch size**: 256
- **Input**: Just conversation context

### Case 2 (With Feedback)
- **Max prompt length**: 2048 tokens (longer due to feedback)
- **Batch size**: 128 (smaller due to longer sequences)
- **Input**: Conversation + Previous Response + Feedback

### Common Settings
- **Max response length**: 768 tokens
- **Learning rate (actor)**: 1e-6
- **Learning rate (critic)**: 1e-5
- **GPUs**: 8 GPUs per node
- **Model**: meta-llama/Llama-3.1-8B-Instruct

## Evaluation Metrics

The evaluation pipeline computes:

1. **Mean reward score** for each case
2. **Statistical significance** (paired t-test, Wilcoxon test)
3. **Effect size** (Cohen's d)
4. **Domain-wise comparison** (code, math, etc.)
5. **Improvement statistics** (how many examples improved/worsened)

## Expected Results

The goal is to test whether providing feedback examples (Case 2) leads to better responses compared to baseline (Case 1).

Look for:
- ✅ Higher mean reward in Case 2
- ✅ Statistical significance (p < 0.05)
- ✅ Positive effect size
- ✅ Consistent improvement across domains

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch sizes in training scripts
- Reduce `max_prompt_length` or `max_response_length`
- Enable more aggressive offloading

### Test Split Mismatch
- Ensure you use the same `--seed` and `--test_size` for both preprocessing runs
- Run `python helpsteer3/verify_test_split_consistency.py` to check

### Reward Model Issues
- The reward model requires GPU and uses bfloat16
- If you see errors, check that the model is properly downloaded
- Ensure you have enough GPU memory for evaluation

## Citation

If you use this pipeline, please cite:

```bibtex
@article{helpsteer3,
  title={HelpSteer3: A Dataset for Training Helpful AI Assistants},
  author={NVIDIA},
  year={2024}
}
```

## Notes

- The test split (500 samples) is deterministically extracted from the validation set using seed=42
- Both cases use the **same test samples** for fair comparison
- The reward model evaluates responses in a **zero-shot** manner (no context about which case)
- Training checkpoints are saved every 5 steps, tested every 2 steps

