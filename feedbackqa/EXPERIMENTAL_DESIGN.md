# Experimental Design: Does Feedback-Augmented Training Help?

## Research Question
**Does showing language models examples of answers with feedback improve their ability to generate high-quality answers?**

## Overview

We compare two training approaches:

| Aspect | Case 1: Question Only | Case 2: With Feedback |
|--------|----------------------|----------------------|
| **Input** | Question | Question + Example Q+A+Feedback |
| **Model Sees** | Just the question to answer | Examples of good answers with feedback |
| **Model Generates** | Answer | Answer (after seeing examples) |
| **RM Scores** | Question + Generated Answer | Question + Generated Answer |
| **Hypothesis** | Learns to answer well from scratch | Learns quality criteria from examples |

## Experimental Design

### Setup

```
Training Data → Train Reward Model (RM)
                        ↓
          RM evaluates answers (0/1 score)
                        ↓
        ┌───────────────┴───────────────┐
        │                               │
   Case 1                          Case 2
(Baseline)                   (Feedback-Augmented)
        │                               │
        ↓                               ↓
  Train with PPO                  Train with PPO
        │                               │
        ↓                               ↓
   Model A                          Model B
```

### Case 1: Question Only (Baseline)

**Training Input:**
```
User: How do I get help finding a job?

[Model generates answer]
```

**RM Scoring:**
```
Input to RM: Question + Generated Answer
Output: Reward score (0-1)
```

**Learning Signal:**
- Model learns from trial and error
- RM provides feedback via reward scores
- No explicit examples of what makes a good answer

### Case 2: Question + Answer + Feedback (Experimental)

**Training Input (Single-Turn):**
```
User: Here is an example of a good answer with feedback:

Example Question: What are COVID-19 symptoms?

Example Answer: Common symptoms include fever, cough, and fatigue...

Feedback: Clear list of symptoms with medical accuracy. (Rating: Excellent)

Now, please answer the following question with similar quality:

Question: How do I get help finding a job?

[Model generates answer]
```

**RM Scoring:**
```
Input to RM: Original Question + Generated Answer (NOT the example!)
Output: Reward score (0-1)
```

**Learning Signal:**
- Model sees examples of good/bad answers
- Model sees what feedback looks like
- Model learns quality criteria explicitly
- Random examples prevent overfitting

### Key Design Choices

1. **Random Feedback Examples**: Each training instance shows a DIFFERENT random Q+A+Feedback example
   - Prevents memorization
   - Exposes model to diverse quality patterns
   - Each example gets different context

2. **RM Scores Generated Answer**: The reward model ONLY scores what the model generates, not the example
   - Fair comparison between cases
   - Model can't game the system by copying examples
   - Tests if learning generalizes

3. **Same Test Procedure**: At inference/test time:
   - Both models receive ONLY the question
   - No feedback or examples at test time
   - Fair comparison of learned capabilities

## Expected Outcomes

### If Case 2 is Better:
- **Result**: Model B generates higher quality answers than Model A
- **Interpretation**: Showing examples with feedback helps models internalize quality criteria
- **Implication**: Few-shot learning with feedback is effective for quality improvement

### If Case 1 is Better:
- **Result**: Model A performs better or equal to Model B
- **Interpretation**: Examples might confuse the model or add noise
- **Implication**: Direct reinforcement learning is sufficient

### If No Difference:
- **Result**: Both models perform similarly
- **Interpretation**: Feedback examples don't provide additional signal beyond RM scores
- **Implication**: RM scores alone are sufficient for learning

## Implementation Details

### Data Preprocessing

**Case 1:**
```python
# preprocess_ppo_case1_question_only.py
prompt = [{"role": "user", "content": question}]
```

**Case 2:**
```python
# preprocess_ppo_case2_with_feedback.py
random_example = sample_random(dataset, current_idx)

# Single-turn prompt with example embedded
prompt_content = (
    f"Here is an example of a good answer with feedback:\n\n"
    f"Example Question: {random_example.question}\n\n"
    f"Example Answer: {random_example.answer}\n\n"
    f"Feedback: {random_example.feedback} (Rating: {random_example.rating})\n\n"
    f"Now, please answer the following question with similar quality:\n\n"
    f"Question: {question}"
)

prompt = [{"role": "user", "content": prompt_content}]
```

### PPO Training Flow

```
1. Sample batch from dataloader
   ↓
2. Model generates response from prompt
   ↓
3. Trained RM scores (Question + Generated Answer)
   ↓
4. Compute advantages using reward scores
   ↓
5. Update policy to maximize rewards
   ↓
6. Repeat
```

### Evaluation

At test time:
```python
# Both models receive the same input format
test_input = [{"role": "user", "content": question}]

# Generate answer
answer = model.generate(test_input)

# Evaluate with RM
score = trained_rm(question + answer)
```

## Files and Scripts

### Data Preprocessing
- `preprocess_ppo_case1_question_only.py`: Create Case 1 dataset
- `preprocess_ppo_case2_with_feedback.py`: Create Case 2 dataset

### Training Scripts
- `run_ppo_case1_question_only.sh`: Train Case 1 model
- `run_ppo_case2_with_feedback.sh`: Train Case 2 model

### Reward Model
- `rm_train.py`: Train the reward model
- `feedback_qa_reward_model/`: Trained RM checkpoint

## Running the Experiment

### Step 1: Train Reward Model
```bash
python feedbackqa/rm_train.py \
    --train_file feedbackqa/feedback_train_rm.json \
    --valid_file feedbackqa/feedback_valid_rm.json \
    --test_file feedbackqa/feedback_test_rm.json \
    --model_name meta-llama/Llama-3.2-3B-Instruct \
    --output_dir ./feedback_qa_reward_model
```

### Step 2: Preprocess Data for Both Cases
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

### Step 3: Train Both Models with PPO
```bash
# Train Case 1 (Baseline)
bash feedbackqa/run_ppo_case1_question_only.sh

# Train Case 2 (With Feedback)
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

### Step 4: Evaluate and Compare
```bash
# Evaluate both models on the same test set
# Compare:
# - Average reward scores
# - Validation metrics
# - Human evaluation
```

## Metrics to Compare

1. **Reward Model Scores**: Average RM score on test set
2. **Validation Metrics**: Loss, perplexity
3. **Human Evaluation**: Quality of generated answers
4. **Training Efficiency**: Convergence speed, sample efficiency
5. **Diversity**: Variety in generated answers

## Expected Timeline

- RM Training: ~2-3 hours (3B model)
- Data Preprocessing: ~5 minutes per case
- PPO Training: ~8-12 hours per case (depends on hardware)
- Evaluation: ~1-2 hours

## Conclusion

This experiment tests a fundamental question in RL for LLMs:

**"Does learning from labeled examples (Q+A+Feedback) improve upon learning from reward signals alone?"**

The answer will inform future approaches to training language models with human feedback.

