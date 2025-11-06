# Updated Experimental Design: Same Question with Previous Answer

## Research Question (Updated)
**"Does showing models previous answers to THE SAME QUESTION with feedback help them generate better answers?"**

## The Corrected Approach

### Case 1: Question Only (Baseline)
```
Input: "How do I get help finding a job?"

Model generates answer
↓
RM scores: Question + Generated Answer
```

### Case 2: Previous Answer to Same Question + Feedback (Experimental)
```
Input:
Here is a previous answer to this question with feedback:

Question: How do I get help finding a job?

Previous Answer: Visit the Jobs Hub website for current vacancies...

Feedback: Includes helpful link to Jobs Hub. (Rating: Excellent)

Now, please answer the same question with similar or better quality:

Question: How do I get help finding a job?

Model generates NEW answer to same question
↓
RM scores: Question + Generated Answer (NOT the previous answer!)
```

## Key Difference from Original Design

### Original (Wrong)
- Showed examples from DIFFERENT questions
- "Example Q: What are COVID symptoms?" → "Now answer: How do I find a job?"
- Model learns general quality patterns across different questions

### Updated (Correct)
- Shows previous answers to the SAME question
- "Previous answer to: How do I find a job?" → "Now answer: How do I find a job?"
- Model learns what makes a good answer to THIS SPECIFIC question

## Why This Makes More Sense

1. **Question-Specific Learning**: Model sees what worked before for THIS question
2. **Direct Comparison**: Can compare previous answer with own attempt
3. **Feedback Relevance**: Feedback is directly applicable to what model needs to do
4. **Realistic Scenario**: Like seeing previous responses before answering

## Data Structure

Your data has multiple entries for same questions:
```json
[
  {
    "question": "How do I get help finding a job?",
    "section_content": "Visit the Jobs Hub...",
    "feedback": "Includes helpful link",
    "rating": "Excellent"
  },
  {
    "question": "How do I get help finding a job?",
    "section_content": "The Australian Government supports...",
    "feedback": "Good overview of sectors",
    "rating": "Good"
  },
  ...
]
```

Case 2 uses this structure:
- When processing entry #2, show entry #1 as "previous answer"
- When processing entry #1, show entry #2 as "previous answer"
- Random selection if multiple answers exist

## Example Training Instance

**Training Instance for Entry #2:**

```
Previous Answer Context:
├─ Question: "How do I get help finding a job?"
├─ Previous Answer: "Visit the Jobs Hub..."
└─ Feedback: "Includes helpful link" (Rating: Excellent)

Now answer:
└─ Question: "How do I get help finding a job?"

Model generates: "The Australian Government supports..."
RM scores: Question + This Generated Answer
```

## Fallback Behavior

If a question appears only once in the dataset:
- No previous answer available
- Falls back to Case 1 format (question only)
- Ensures all data can be used

## Statistics to Track

When preprocessing, you'll see:
```
Dataset Statistics:
  Total examples: 32,402
  Examples with previous answer: ~28,000 (85%)
  Examples without previous answer: ~4,000 (15%)
```

This tells you:
- How many questions have multiple answers
- How much "feedback augmentation" you're actually getting

## Expected Outcome

### If Case 2 Wins:
**Interpretation**: Seeing previous answers to the SAME question helps
- Model learns question-specific quality criteria
- Feedback directly teaches what works for this question
- Better than learning from scratch

### If Case 1 Wins:
**Interpretation**: Previous answers don't help or confuse
- Model might copy previous answer instead of innovating
- RM scoring alone is sufficient
- Direct exploration works better

## Comparison Table

| Aspect | Case 1 | Case 2 |
|--------|--------|--------|
| **Input** | Question only | Previous answer to SAME question + feedback + question |
| **Context** | No examples | Previous answer example |
| **Learning** | Trial & error | Learn from what worked before |
| **Question** | Same as in training | SAME question repeated |
| **Feedback Relevance** | N/A | Directly applicable |
| **RM Scoring** | Q + Generated A | Q + Generated A (not previous!) |

## Why This Is Better

1. **More Focused Learning**: Question-specific rather than general
2. **Uses Your Data Structure**: Multiple answers per question
3. **Clearer Hypothesis**: Does seeing previous attempts help?
4. **Practical Application**: Models can learn from their own history

## Running The Updated Experiment

```bash
# Preprocess with updated logic
python feedbackqa/preprocess_ppo_case2_with_feedback.py

# Check statistics
python feedbackqa/inspect_preprocessed_data.py

# Train
bash feedbackqa/run_ppo_case2_with_feedback.sh
```

You'll see output like:
```
Processing 32,402 examples for Case 2...
✓ Found previous answers for 28,156 examples (86.9%)
✓ Using question-only for 4,246 examples (13.1%)

Example prompt:
  Here is a previous answer to this question with feedback:

  Question: How do I get help finding a job?

  Previous Answer: Visit the Jobs Hub website...

  Feedback: Includes helpful link. (Rating: Excellent)

  Now, please answer the same question with similar or better quality:

  Question: How do I get help finding a job?
```

This is now the correct experimental design! 🎯

