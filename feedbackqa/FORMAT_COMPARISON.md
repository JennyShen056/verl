# Format Comparison: Case 1 vs Case 2

## Visual Comparison

### Case 1: Question Only (Baseline)
```
┌─────────────────────────────────────────────┐
│              Prompt Format                   │
│  (Like gsm8k.py - single-turn)              │
├─────────────────────────────────────────────┤
│                                              │
│  prompt = [                                  │
│      {                                       │
│          "role": "user",                     │
│          "content": "How do I find a job?"   │
│      }                                       │
│  ]                                           │
│                                              │
└─────────────────────────────────────────────┘
                    ↓
              Model generates
                    ↓
         "Visit job websites..."
                    ↓
         RM scores: Q + Generated Answer
```

### Case 2: With Feedback (Experimental - Single-Turn)
```
┌──────────────────────────────────────────────────────┐
│                  Prompt Format                        │
│        (Like gsm8k.py - single-turn)                 │
├──────────────────────────────────────────────────────┤
│                                                       │
│  prompt_content = f"""                                │
│  Here is an example of a good answer with feedback:  │
│                                                       │
│  Example Question: What are COVID symptoms?          │
│                                                       │
│  Example Answer: Fever, cough, fatigue...            │
│                                                       │
│  Feedback: Clear symptom list. (Rating: Excellent)   │
│                                                       │
│  Now, please answer the following question           │
│  with similar quality:                               │
│                                                       │
│  Question: How do I find a job?                      │
│  """                                                  │
│                                                       │
│  prompt = [                                           │
│      {                                                │
│          "role": "user",                              │
│          "content": prompt_content                    │
│      }                                                │
│  ]                                                    │
│                                                       │
└──────────────────────────────────────────────────────┘
                    ↓
              Model generates
                    ↓
  "Visit Jobs Hub for vacancies. Consider sectors
   like health, logistics, retail..."
                    ↓
         RM scores: Q + Generated Answer
    (NOTE: RM scores the ORIGINAL question + generated answer,
     NOT the example question!)
```

## Key Points

### Both Use Single-Turn Format
✅ **Case 1**: One user message with just the question  
✅ **Case 2**: One user message with example + question embedded

This is similar to how `gsm8k.py` works:
```python
# gsm8k.py example
question = question_raw + " " + instruction_following
prompt = [{"role": "user", "content": question}]
```

### Why Single-Turn for Case 2?

1. **Simpler for model**: All context in one message
2. **Consistent with verl examples**: gsm8k.py uses single-turn
3. **Easier to process**: Model sees everything at once
4. **Clear structure**: Example → Instruction → Question

### What Gets Scored?

**Case 1:**
```
RM Input: "How do I find a job?" + "Visit job websites..."
RM Output: 0.65
```

**Case 2:**
```
RM Input: "How do I find a job?" + "Visit Jobs Hub for vacancies..."
          ^                        ^
          |                        |
      ORIGINAL question     Generated answer
      (NOT the example!)

RM Output: 0.85
```

**Important**: The example Q+A+Feedback is **only for context during training**. The RM only scores the model's generated answer to the ORIGINAL question.

## Example Data Flow

### Preprocessing
```python
# Input JSON
{
    "question": "How do I find a job?",
    "section_content": "Visit Jobs Hub for vacancies...",
    "feedback": "Includes helpful link",
    "rating": "Excellent"
}

# After Case 1 preprocessing
{
    "prompt": [
        {"role": "user", "content": "How do I find a job?"}
    ]
}

# After Case 2 preprocessing
{
    "prompt": [
        {
            "role": "user", 
            "content": "Here is an example...\n\n
                       Example Question: What are COVID symptoms?\n\n
                       Example Answer: Fever, cough...\n\n
                       Feedback: Clear list. (Rating: Excellent)\n\n
                       Now answer: How do I find a job?"
        }
    ]
}
```

### Training
```python
# PPO receives the preprocessed batch

# Case 1
batch["prompt"] = [{"role": "user", "content": "How do I find a job?"}]
→ Model generates answer
→ RM scores: Q + Generated Answer

# Case 2
batch["prompt"] = [{"role": "user", "content": "Here is an example..."}]
→ Model sees example context
→ Model generates answer
→ RM scores: Original Q + Generated Answer (not example!)
```

### Testing
```python
# At test time, BOTH models receive:
test_input = [{"role": "user", "content": "How do I find a job?"}]

# No feedback examples at test time!
# This tests what the model learned during training.
```

## Random Example Selection

Each training instance shows a **different random example**:

```python
Training Step 1:
  Example shown: COVID symptoms
  Actual question: Job search

Training Step 2:
  Example shown: Visa requirements  
  Actual question: Travel info

Training Step 3:
  Example shown: Job search
  Actual question: COVID symptoms

# Random, not fixed!
```

This prevents:
- ❌ Memorizing specific examples
- ❌ Overfitting to certain patterns

This encourages:
- ✅ Learning general quality principles
- ✅ Abstracting feedback patterns
- ✅ Generalizing to new questions

## Summary

| Aspect | Case 1 | Case 2 |
|--------|--------|--------|
| **Format** | Single-turn | Single-turn |
| **Content** | Question only | Example + Question |
| **Context Length** | Short (~50 tokens) | Long (~300-500 tokens) |
| **RM Scoring** | Q + Generated A | Original Q + Generated A |
| **Training Signal** | RM scores only | RM scores + examples |
| **Test Format** | Question only | Question only |

Both are single-turn, making them compatible with verl's PPO pipeline and similar to the gsm8k.py format!

