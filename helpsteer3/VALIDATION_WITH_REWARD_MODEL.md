# Validation Metrics with Reward Model in VERL

## The Issue

When using a **reward model** (instead of a reward function), validation metrics are **NOT automatically computed** during PPO training. This is different from using rule-based reward functions where validation works out of the box.

## Why Validation Doesn't Work with Reward Models

Looking at the VERL code (`verl/trainer/ppo/ray_trainer.py`), the validation logic requires a `val_reward_fn` to be set:

```python
def _validate(self):
    ...
    # evaluate using reward_function
    if self.val_reward_fn is None:
        raise ValueError("val_reward_fn must be provided for validation.")
    result = self.val_reward_fn(test_batch, return_dict=True)
    reward_tensor = result["reward_tensor"]
    scores = reward_tensor.sum(-1).cpu().tolist()
```

**The problem**: When you use `reward_model.enable=True`, the system sets up a reward model worker for **training**, but it does NOT automatically create a `val_reward_fn` for validation.

## Current State

### What You Have Now

Your current training configuration:
```bash
reward_model.enable=True
reward_model.model.path="$HOME/models/llama3_help_rm"
trainer.test_freq=2  # This setting is IGNORED without val_reward_fn
```

### What's Happening

1. ✅ **Training** works fine - reward model scores training examples
2. ❌ **Validation** is skipped - no `val_reward_fn` configured
3. ❌ **No validation metrics** appear in W&B

The `test_freq=2` setting tells the trainer to validate every 2 steps, but since there's no `val_reward_fn`, validation is silently skipped.

## Solution Options

### Option 1: Use Post-Training Evaluation (RECOMMENDED)

**This is what you're already doing** with `run_full_evaluation.sh`! Instead of validation during training:

1. Train both models completely
2. Use your evaluation pipeline to test on a held-out test set
3. Compare final checkpoint performance

**Advantages**:
- More thorough evaluation
- Can test multiple checkpoints
- Better statistical analysis
- Already implemented in your pipeline

**Disadvantages**:
- Don't see validation metrics during training
- Can't do early stopping based on validation

### Option 2: Create a Custom Validation Reward Function

You would need to create a custom reward function that wraps your reward model for validation:

```python
# custom_val_reward.py
import torch
from transformers import AutoTokenizer, pipeline

def compute_val_reward(data, return_dict=True):
    """Custom validation reward function using the reward model"""
    
    # Load reward model (cache this in practice)
    rm_tokenizer = AutoTokenizer.from_pretrained("Jennny/llama3_help_rm")
    if rm_tokenizer.pad_token is None:
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        rm_tokenizer.pad_token_id = rm_tokenizer.eos_token_id
    
    rm_pipe = pipeline(
        "sentiment-analysis",
        model="Jennny/llama3_help_rm",
        device=0,
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    
    # Extract prompts and responses from data
    batch = data.batch
    prompts = batch["prompts"]  # or however prompts are stored
    responses = batch["responses"]
    
    # Create chats for reward model
    chats = []
    for prompt, response in zip(prompts, responses):
        chat = prompt.copy()  # Assuming prompt is already a list of messages
        chat.append({"role": "assistant", "content": response})
        chats.append(chat)
    
    # Tokenize
    texts = [
        rm_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        .replace(rm_tokenizer.bos_token, "")
        for chat in chats
    ]
    
    # Get rewards
    outputs = rm_pipe(texts, top_k=None, function_to_apply="none", batch_size=8)
    rewards = torch.tensor([output[0]["score"] for output in outputs])
    
    if return_dict:
        return {
            "reward_tensor": rewards.unsqueeze(-1),  # Shape: (batch_size, 1)
            "reward_extra_info": {}
        }
    return rewards.unsqueeze(-1)
```

Then use it:
```bash
python3 -m verl.trainer.main_ppo \
    ...
    reward_model.enable=True \
    reward_model.model.path="$HOME/models/llama3_help_rm" \
    custom_reward_function.path="custom_val_reward.py" \
    custom_reward_function.name="compute_val_reward" \
    trainer.test_freq=2
```

**Challenges**:
- Performance: Loading reward model for each validation
- Memory: Need to keep reward model in memory
- Complexity: Need to handle data format carefully

### Option 3: Monitor Training Rewards Only

Track the **training reward scores** instead of validation:

```bash
# Your current config already logs these
critic/score/mean: 0.93359375  # Mean reward during training
critic/score/max: 8.0           # Max reward
critic/score/min: -6.90625      # Min reward
```

**What these tell you**:
- How well the model is doing on training data
- Whether rewards are improving over time
- If training is stable

**In W&B**, look for:
- `critic/rewards/mean` - Average reward on training batches
- `critic/score/mean` - Same as rewards/mean
- `actor/entropy` - Model exploration (should decrease slowly)
- `critic/vf_loss` - Value function loss

## Recommended Approach for HelpSteer3

Based on your setup, I recommend **Option 1** (post-training evaluation):

### During Training
1. Monitor training rewards: `critic/rewards/mean`
2. Watch for stability: gradual improvement without collapse
3. Check for overfitting: response length hitting max (765/768 - you're at the limit!)

### After Training  
1. Use `run_full_evaluation.sh` to evaluate checkpoints
2. Compare multiple checkpoints (steps 115, 201, etc.)
3. Get rigorous statistical comparison

## Metrics You Can Monitor During Training

Even without validation, you have rich training metrics:

### Reward Metrics
- `critic/rewards/mean` - Average reward
- `critic/rewards/max/min` - Reward range
- `critic/score/mean` - Same as rewards

### Policy Metrics
- `actor/pg_loss` - Policy gradient loss
- `actor/entropy` - Exploration (should decrease gradually)
- `actor/ppo_kl` - KL divergence (monitor for collapse)
- `actor/grad_norm` - Gradient magnitude

### Response Quality
- `response_length/mean` - Average response length (yours: 765/768)
- `response_length/clip_ratio` - % hitting max length (yours: 99.6% - concerning!)
- `response/aborted_ratio` - % of aborted generations

### Value Function
- `critic/vf_loss` - Value function training loss
- `critic/vf_explained_var` - How well critic predicts returns

## ⚠️ Critical Issue in Your Training

From your logs:
```
response_length/mean:765.19140625
response_length/max:768.0
response_length/clip_ratio:0.99609375  # 99.6% hitting max!
```

**Almost all responses are hitting the max length!** This suggests:
1. Model wants to generate longer responses
2. Being cut off at 768 tokens
3. May be affecting reward quality

### Recommended Fix:
```bash
# Increase max_response_length
data.max_response_length=1024  # or even 1536
```

Test what length you actually need using the analysis script:
```bash
python helpsteer3/analyze_dataset_lengths.py --splits validation --max_samples 500
```

## Summary

**For HelpSteer3 experiments:**

1. ✅ **Keep using post-training evaluation** - your current approach is good
2. ✅ **Monitor training rewards** in W&B - `critic/rewards/mean`  
3. ⚠️ **Fix response length** - increase to 1024 or 1536 tokens
4. ✅ **Use checkpoints** - evaluate multiple steps (115, 201, etc.)
5. ✅ **Compare final results** - your evaluation pipeline does this well

**Validation during training is NOT critical** when you have:
- Good training reward monitoring
- Post-training evaluation pipeline
- Multiple checkpoints to choose from

Your current setup is actually quite reasonable for research!

