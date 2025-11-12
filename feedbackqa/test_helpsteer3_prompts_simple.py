"""
Simple test script to visualize HelpSteer3 prompt formats without running inference.
Shows the exact prompts that will be sent to the model.

Usage:
  python test_helpsteer3_prompts_simple.py  # Just show prompts
  python test_helpsteer3_prompts_simple.py --generate  # Generate responses (requires GPU)
"""

import argparse
import random
from typing import Dict, List

from datasets import load_dataset


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


def print_separator(title: str, char: str = "="):
    """Print a nice separator"""
    print(f"\n{char*80}")
    print(f"  {title}")
    print(f"{char*80}\n")


def print_messages(messages: List[Dict], title: str):
    """Pretty print message list"""
    print_separator(title, "─")
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        print(f"[Message {i+1} - {role}]")
        print(content)
        if i < len(messages) - 1:
            print("\n" + "─"*40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test HelpSteer3 prompt formats")
    parser.add_argument("--num_examples", type=int, default=2, help="Number of examples to show")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--generate", action="store_true", help="Actually generate responses (requires GPU)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                       help="Model to use for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print_separator("Loading HelpSteer3 Dataset")
    dataset = load_dataset("nvidia/HelpSteer3", "feedback", split="train")
    print(f"Loaded {len(dataset)} examples")
    
    # Sample examples
    sample_indices = random.sample(range(len(dataset)), args.num_examples)
    print(f"Selected examples at indices: {sample_indices}")
    
    # Load model if generating
    model = None
    tokenizer = None
    if args.generate:
        print_separator("Loading Model and Tokenizer")
        print(f"Model: {args.model}")
        
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Model loaded successfully")
    
    # Process each example
    for example_num, idx in enumerate(sample_indices, 1):
        example = dataset[idx]
        
        print_separator(f"EXAMPLE {example_num}/{args.num_examples} (Index: {idx})")
        
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
        
        print(f"Domain: {domain} | Language: {language} | Using Response{chosen_id}")
        
        # Show original conversation
        print(f"\n📋 Original Conversation:")
        print("─" * 80)
        for i, turn in enumerate(context):
            print(f"{turn['role'].upper()}: {turn['content'][:200]}{'...' if len(turn['content']) > 200 else ''}")
        
        # Show chosen response and feedback
        print(f"\n📝 Chosen Previous Response (Response{chosen_id}):")
        print("─" * 80)
        print(chosen_response[:400] + ("..." if len(chosen_response) > 400 else ""))
        
        print(f"\n💬 Feedback on Previous Response:")
        print("─" * 80)
        for fb in chosen_feedback[:3]:  # Show first 3 feedback items
            print(f"  • {fb}")
        if len(chosen_feedback) > 3:
            print(f"  ... and {len(chosen_feedback) - 3} more")
        
        # Create prompts
        case1_messages = create_case1_prompt(context)
        case2_messages = create_case2_prompt(context, chosen_response, chosen_feedback)
        
        # Show Case 1 prompt
        print_messages(case1_messages, "🔷 CASE 1 PROMPT: Question Only (Baseline)")
        
        print(f"\nCase 1 prompt stats:")
        print(f"  - Number of messages: {len(case1_messages)}")
        print(f"  - Total characters: {sum(len(m['content']) for m in case1_messages)}")
        
        # Show Case 2 prompt
        print_messages(case2_messages, "🔶 CASE 2 PROMPT: With Previous Response + Feedback")
        
        print(f"\nCase 2 prompt stats:")
        print(f"  - Number of messages: {len(case2_messages)}")
        print(f"  - Total characters: {sum(len(m['content']) for m in case2_messages)}")
        
        # Generate if requested
        if args.generate and model is not None:
            print_separator("Generating Responses")
            
            import torch
            
            def generate(messages):
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        temperature=0.7,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            print("🤖 Case 1 Response (Baseline):")
            print("─" * 80)
            case1_response = generate(case1_messages)
            print(case1_response)
            
            print(f"\n🎓 Case 2 Response (With Feedback):")
            print("─" * 80)
            case2_response = generate(case2_messages)
            print(case2_response)
            
            print(f"\n📊 Comparison:")
            print(f"  Case 1 length: {len(case1_response)} chars")
            print(f"  Case 2 length: {len(case2_response)} chars")
            print(f"  Different: {'✓ Yes' if case1_response != case2_response else '✗ No (identical)'}")
        
        print("\n" + "="*80 + "\n")
    
    print_separator("Summary")
    print(f"Showed {args.num_examples} examples")
    if args.generate:
        print("\n💡 Key things to observe:")
        print("  1. Does Case 2 produce higher quality responses?")
        print("  2. Does Case 2 learn from the feedback?")
        print("  3. Are Case 2 responses more detailed/accurate?")
        print("  4. Does Case 2 avoid issues mentioned in the feedback?")
    else:
        print("\n💡 To generate actual model responses, run with --generate flag")
        print("   (Requires GPU and model access)")


if __name__ == "__main__":
    main()

