# Raw Prompt KeyError Fix

## 🔍 **The Error**

```python
KeyError: 'raw_prompt'
File "/workspace/verl/verl/workers/fsdp_workers.py", line 1879, in compute_rm_score
    rm_data = self._switch_chat_template(data)
File "/workspace/verl/verl/workers/fsdp_workers.py", line 1814, in _switch_chat_template
    if not isinstance(data.non_tensor_batch["raw_prompt"][i], list | np.ndarray):
KeyError: 'raw_prompt'
```

## 📊 **Root Cause**

The error occurs because the reward model worker tries to **switch chat templates** between the policy model and reward model.

### The Flow:

1. **Policy model** generates responses using `meta-llama/Llama-3.1-8B-Instruct`
2. **Reward model** (`FsfairX-LLaMA3-RM-v0.1`) needs to score these responses
3. By default, verl's config has:
   ```yaml
   reward_model.model.input_tokenizer: ${actor_rollout_ref.model.path}
   ```
4. This tells verl: "The input is tokenized with policy model's tokenizer, but reward model uses a different tokenizer"
5. So verl tries to:
   - Decode the tokens back to text using policy tokenizer
   - Re-tokenize with reward model tokenizer
6. **But**: This requires the `raw_prompt` field (the original conversation) in the data
7. **Problem**: Your preprocessed data doesn't include `raw_prompt`

## ✅ **The Fix**

Since both models (Llama-3.1-8B-Instruct and FsfairX-LLaMA3-RM-v0.1) are based on Llama-3 and use **the same chat template**, we can disable template switching:

```bash
reward_model.model.input_tokenizer=null
```

This tells verl: "Don't switch tokenizers, use the tokens as-is"

### Updated Scripts:

**Case 1 (line 73):**
```bash
reward_model.model.path="$HOME/models/FsfairX-LLaMA3-RM-v0.1" \
reward_model.model.input_tokenizer=null \  # ← NEW
reward_model.model.use_remove_padding=True \
```

**Case 2 (line 70):**
```bash
reward_model.model.path="$TRAINED_RM_PATH" \
reward_model.model.input_tokenizer=null \  # ← NEW
reward_model.model.use_remove_padding=True \
```

## 🤔 **When to Use Each Approach**

### Disable Template Switching (`input_tokenizer=null`)

**✅ Use when:**
- Policy model and reward model use **the same chat template**
- Both are from the same model family (e.g., Llama-3.x)
- You want simpler, faster processing
- Your preprocessed data doesn't include `raw_prompt`

**Pros:**
- No re-tokenization overhead
- Faster inference
- Simpler data format
- No need for `raw_prompt` field

**Cons:**
- Won't work if chat templates differ

### Enable Template Switching (`input_tokenizer=<path>`)

**✅ Use when:**
- Policy and reward models use **different chat templates**
- E.g., Policy is Llama-3, Reward is GPT-2 format
- You need template conversion

**Requires:**
- `raw_prompt` field in preprocessed data
- More complex preprocessing
- Re-tokenization overhead

## 📝 **How to Add `raw_prompt` (If Needed)**

If you ever need template switching, modify your preprocessing script:

```python
def make_map_fn(split: str, data_source: str = "feedback_qa"):
    def process_fn(example: Dict, idx: int) -> Dict:
        question = example["question"]
        
        prompt_messages = [
            {"role": "user", "content": question}
        ]
        
        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "raw_prompt": prompt_messages,  # ← ADD THIS
            "ability": "qa_generation",
            # ... rest of the data
        }
        return data
    
    return process_fn
```

But for your case, **this isn't needed** since you're disabling template switching.

## 🎯 **Why This Works**

Both your models are Llama-3 based:

| Model | Type | Chat Template |
|-------|------|---------------|
| `meta-llama/Llama-3.1-8B-Instruct` | Policy | Llama-3 format |
| `FsfairX-LLaMA3-RM-v0.1` | Reward | Llama-3 format |

Since they use the **same template**, no conversion is needed! The tokens from the policy model can be directly scored by the reward model.

## 🔧 **Summary of All Fixes**

| Issue | Fix | Line |
|-------|-----|------|
| Empty validation metrics | `trainer.val_before_train=False` | 84 (Case 1), 81 (Case 2) |
| Missing `raw_prompt` | `reward_model.model.input_tokenizer=null` | 73 (Case 1), 70 (Case 2) |

## ✅ **Current Status**

Both training scripts now have:
- ✅ Skip initial validation
- ✅ Disable chat template switching
- ✅ Use pre-trained reward model
- ✅ Proper error checking
- ✅ Correct batch sizes and lengths

**Ready to train!** 🚀

```bash
bash feedbackqa/run_ppo_case1_question_only.sh
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

## 📚 **Related Documentation**

- `VALIDATION_FIX.md` - Explains the validation metrics fix
- `TRAINING_WORKFLOW.md` - Complete training workflow
- `ANALYSIS_RESULTS.md` - Prompt/response length analysis

---

**Note**: The `input_tokenizer` setting is in the config file at:
```yaml
# verl/trainer/config/reward_model/reward_model.yaml
model:
  input_tokenizer: ${actor_rollout_ref.model.path}  # Default: enables switching
```

By setting it to `null` in the training script, we override this default.

