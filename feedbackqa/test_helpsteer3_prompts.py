"""
Test script to compare HelpSteer3 prompt formats:
- Case 1: Question only (baseline)
- Case 2: With previous response and feedback

Uses meta-llama/Llama-3.1-8B-Instruct to generate responses.
"""

import random
from typing import Dict, List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_conversation(context: List[Dict]) -> str:
    """Convert conversation context list to a formatted string"""
    formatted = []
    for turn in context:
        role = turn["role"]
        content = turn["content"]
        if role == "user":
            formatted.append(f"User: {content}")
        elif role == "assistant":
            formatted.append(f"Assistant: {content}")
    return "\n\n".join(formatted)


def format_feedback_list(feedback_list: List[str]) -> str:
    """Convert feedback list to a formatted string"""
    if not feedback_list:
        return "No specific feedback provided."
    return "\n".join([f"- {fb}" for fb in feedback_list])


def create_case1_prompt(context: List[Dict]) -> List[Dict]:
    """Case 1: Question only - just the conversation context"""
    return context.copy()


def create_case2_prompt(context: List[Dict], chosen_response: str, chosen_feedback: List[str]) -> List[Dict]:
    """Case 2: With previous response and feedback"""
    formatted_context = format_conversation(context)
    formatted_feedback = format_feedback_list(chosen_feedback)
    
    # Single-turn prompt with previous response to SAME context
    prompt_content = (
        f"Here is a previous response to this conversation with feedback:\n\n"
        f"Conversation:\n{formatted_context}\n\n"
        f"Previous Response: {chosen_response}\n\n"
        f"Feedback: {formatted_feedback}\n\n"
        f"Now, please respond to the same conversation:\n\n"
        f"Conversation:\n{formatted_context}"
    )
    
    return [{"role": "user", "content": prompt_content}]


def generate_response(model, tokenizer, messages: List[Dict], max_new_tokens: int = 512) -> str:
    """Generate response using the model"""
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated tokens (exclude prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def print_separator(title: str):
    """Print a nice separator"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def main():
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    NUM_EXAMPLES = 3
    SEED = 42
    
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    print_separator("Loading HelpSteer3 Dataset")
    dataset = load_dataset("nvidia/HelpSteer3", "feedback", split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Sample a few examples
    sample_indices = random.sample(range(len(dataset)), NUM_EXAMPLES)
    print(f"Selected examples at indices: {sample_indices}")
    
    print_separator("Loading Model and Tokenizer")
    print(f"Model: {MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Model loaded on device: {model.device}")
    print(f"Model dtype: {model.dtype}")
    
    # Process each example
    for example_num, idx in enumerate(sample_indices, 1):
        example = dataset[idx]
        
        print_separator(f"Example {example_num}/{NUM_EXAMPLES} (Index: {idx})")
        
        # Extract data
        context = example["context"]
        response1 = example.get("response1", "")
        response2 = example.get("response2", "")
        feedback1 = example.get("feedback1", [])
        feedback2 = example.get("feedback2", [])
        domain = example.get("domain", "N/A")
        language = example.get("language", "N/A")
        
        # Randomly choose response1 or response2
        if random.random() < 0.5:
            chosen_response = response1
            chosen_feedback = feedback1
            chosen_id = 1
        else:
            chosen_response = response2
            chosen_feedback = feedback2
            chosen_id = 2
        
        print(f"Domain: {domain}")
        print(f"Language: {language}")
        print(f"Chosen response{chosen_id} for Case 2")
        
        # Display the original conversation
        print(f"\n--- Original Conversation ---")
        formatted_conv = format_conversation(context)
        print(formatted_conv[:500] + ("..." if len(formatted_conv) > 500 else ""))
        
        # Display the chosen response and feedback
        print(f"\n--- Chosen Previous Response (Response{chosen_id}) ---")
        print(chosen_response[:300] + ("..." if len(chosen_response) > 300 else ""))
        
        print(f"\n--- Feedback on Previous Response ---")
        formatted_feedback = format_feedback_list(chosen_feedback)
        print(formatted_feedback[:300] + ("..." if len(formatted_feedback) > 300 else ""))
        
        # Create prompts
        case1_messages = create_case1_prompt(context)
        case2_messages = create_case2_prompt(context, chosen_response, chosen_feedback)
        
        # Generate responses
        print(f"\n{'─'*80}")
        print("🤖 CASE 1: Question Only (Baseline)")
        print(f"{'─'*80}")
        case1_response = generate_response(model, tokenizer, case1_messages)
        print(case1_response)
        
        print(f"\n{'─'*80}")
        print("🎓 CASE 2: With Previous Response + Feedback")
        print(f"{'─'*80}")
        case2_response = generate_response(model, tokenizer, case2_messages)
        print(case2_response)
        
        # Quick comparison
        print(f"\n{'─'*80}")
        print("📊 COMPARISON")
        print(f"{'─'*80}")
        print(f"Case 1 length: {len(case1_response)} chars")
        print(f"Case 2 length: {len(case2_response)} chars")
        
        # Check if responses are different
        if case1_response == case2_response:
            print("⚠️  Responses are identical!")
        else:
            print("✓ Responses are different")
        
        print("\n" + "="*80 + "\n")
    
    print_separator("Testing Complete")
    print(f"Processed {NUM_EXAMPLES} examples")
    print("\nObservations to look for:")
    print("1. Does Case 2 produce higher quality responses?")
    print("2. Does Case 2 learn from the feedback?")
    print("3. Are Case 2 responses more detailed/accurate?")
    print("4. Does Case 2 avoid issues mentioned in the feedback?")


if __name__ == "__main__":
    with open("helpsteer3_results.txt", "w", encoding="utf8") as f:
        with redirect_stdout(f):
            main()