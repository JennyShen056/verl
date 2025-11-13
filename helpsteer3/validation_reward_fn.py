"""
Custom validation reward function for HelpSteer3 that uses the reward model.
This enables validation metrics during PPO training even when using model-based rewards.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Global cache for model and tokenizer to avoid reloading
_RM_MODEL = None
_RM_TOKENIZER = None
_RM_DEVICE = None


def _load_reward_model(model_path: str = "Jennny/llama3_help_rm"):
    """Load and cache the reward model and tokenizer"""
    global _RM_MODEL, _RM_TOKENIZER, _RM_DEVICE
    
    if _RM_MODEL is None:
        print(f"Loading validation reward model from {model_path}...")
        
        _RM_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad_token if not set
        if _RM_TOKENIZER.pad_token is None:
            _RM_TOKENIZER.pad_token = _RM_TOKENIZER.eos_token
            _RM_TOKENIZER.pad_token_id = _RM_TOKENIZER.eos_token_id
        
        _RM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        _RM_MODEL = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=_RM_DEVICE,
        )
        _RM_MODEL.eval()
        
        print(f"Validation reward model loaded on {_RM_DEVICE}")
    
    return _RM_MODEL, _RM_TOKENIZER, _RM_DEVICE


def compute_score(data, return_dict=True, model_path: str = "Jennny/llama3_help_rm", **kwargs):
    """
    Compute validation rewards using the reward model.
    
    Args:
        data: DataProto object containing batch data
        return_dict: Whether to return a dictionary (required by VERL)
        model_path: Path to the reward model
        **kwargs: Additional arguments (ignored)
    
    Returns:
        Dictionary with reward_tensor if return_dict=True, else just tensor
    """
    # Load model (cached after first call)
    rm_model, rm_tokenizer, device = _load_reward_model(model_path)
    
    batch = data.batch
    
    # Extract prompts and responses
    # The prompt comes from the data, and we need to reconstruct the conversation
    if "prompt_for_rm" in batch:
        # Use the stored prompt for RM if available
        prompts = batch["prompt_for_rm"]
    elif "prompt" in data.non_tensor_batch:
        prompts = data.non_tensor_batch["prompt"]
    else:
        raise ValueError("No prompt found in batch data")
    
    # Get responses (these are token IDs)
    response_ids = batch["responses"]
    
    # Decode responses
    responses = []
    for resp_ids in response_ids:
        # Remove padding tokens
        resp_ids_clean = resp_ids[resp_ids != rm_tokenizer.pad_token_id]
        response_text = rm_tokenizer.decode(resp_ids_clean, skip_special_tokens=True)
        responses.append(response_text)
    
    # Create conversations for reward model
    chats = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # Prompt is either a list of messages or needs to be converted
        if isinstance(prompt, list):
            chat = prompt.copy()
        else:
            # Assume it's a string, convert to user message
            chat = [{"role": "user", "content": prompt}]
        
        # Add the model's response
        chat.append({"role": "assistant", "content": response})
        chats.append(chat)
    
    # Prepare texts for reward model
    texts = []
    for chat in chats:
        text = rm_tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=False
        )
        # Remove BOS token as per reward model requirements
        text = text.replace(rm_tokenizer.bos_token, "")
        texts.append(text)
    
    # Tokenize
    inputs = rm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=4096,  # Adjust based on your needs
    ).to(device)
    
    # Get rewards
    with torch.no_grad():
        outputs = rm_model(**inputs)
        # The reward is typically in the logits, shape: (batch_size, num_classes)
        # For regression models, it's usually (batch_size, 1)
        # For classification models, we might need to take a specific class
        logits = outputs.logits
        
        if logits.shape[-1] == 1:
            # Regression model - direct score
            rewards = logits.squeeze(-1)
        else:
            # Classification model - take the positive class score
            # This assumes class 1 is the "good" class
            rewards = logits[:, 1] if logits.shape[-1] == 2 else logits[:, -1]
    
    # Move back to CPU and ensure correct shape for VERL
    # VERL expects shape: (batch_size, sequence_length) or (batch_size, 1)
    # For validation, we typically use sentence-level rewards
    reward_tensor = rewards.cpu().unsqueeze(-1)  # Shape: (batch_size, 1)
    
    if return_dict:
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": {
                "val_reward_mean": float(reward_tensor.mean()),
                "val_reward_std": float(reward_tensor.std()),
                "val_reward_max": float(reward_tensor.max()),
                "val_reward_min": float(reward_tensor.min()),
            }
        }
    
    return reward_tensor


# For compatibility, also export under the default name
compute_val_reward = compute_score

