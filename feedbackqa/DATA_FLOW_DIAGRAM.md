# Reward Model Training: Data Flow Diagram

## 📊 Complete Data Transformation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Raw Input Data (JSON)                                   │
└─────────────────────────────────────────────────────────────────┘

{
  "question": "How do I get help finding a job?",
  "section_content": "In this rapidly changing jobs market, it can be 
                      difficult to know where to start when looking for 
                      employment. The good news is that there are many 
                      resources available to help you...",
  "label": 1  // 1 = good answer, 0 = bad answer
}

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Format with Chat Template                               │
│ (format_with_chat_template function)                            │
└─────────────────────────────────────────────────────────────────┘

messages = [
  {"role": "user", "content": "How do I get help finding a job?"},
  {"role": "assistant", "content": "In this rapidly changing..."}
]

text = tokenizer.apply_chat_template(messages, tokenize=False)

Result:
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

How do I get help finding a job?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

In this rapidly changing jobs market, it can be difficult to know 
where to start when looking for employment. The good news is that 
there are many resources available to help you...<|eot_id|>

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Tokenization                                            │
│ (tokenize_function)                                             │
└─────────────────────────────────────────────────────────────────┘

tokenizer(text, max_length=1024, truncation=True)

{
  "input_ids": [
    128000,  // <|begin_of_text|>
    128006,  // <|start_header_id|>
    882,     // "user"
    ...      // question tokens
    128009,  // <|eot_id|>
    128006,  // <|start_header_id|>
    78191,   // "assistant"
    ...      // answer tokens
    128009   // <|eot_id|> (EOS token)
  ],
  "attention_mask": [1, 1, 1, ..., 1],
  "labels": 1.0  // Float (not int!) for verl compatibility
}

Length: 287 tokens

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Batching (DataLoader)                                   │
└─────────────────────────────────────────────────────────────────┘

Batch of 4 examples:

input_ids shape:      [4, 512]  // batch_size=4, padded to max_len=512
attention_mask shape: [4, 512]
labels shape:         [4]       // [1.0, 0.0, 1.0, 1.0]

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Model Forward Pass                                      │
│ (AutoModelForTokenClassification with num_labels=1)             │
└─────────────────────────────────────────────────────────────────┘

Input:  [4, 512] token IDs
        ↓
    Embedding Layer
        ↓
    Transformer Layers (32 layers for Llama-3.1-8B)
        ↓
    Classification Head (outputs 1 score per token)
        ↓
Output: [4, 512, 1]  // One score for each token in each sequence

Example output values:
[
  [[0.12], [0.15], ..., [0.87]],  // Sequence 1, EOS score: 0.87
  [[0.09], [0.11], ..., [-0.23]], // Sequence 2, EOS score: -0.23
  [[0.14], [0.18], ..., [0.91]],  // Sequence 3, EOS score: 0.91
  [[0.10], [0.13], ..., [0.45]]   // Sequence 4, EOS score: 0.45
]

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Extract EOS Token Scores                                │
│ (for loss computation and evaluation)                           │
└─────────────────────────────────────────────────────────────────┘

Take the last token (EOS) score from each sequence:

eos_scores = output[:, -1, 0]  // Shape: [4]

Values: [0.87, -0.23, 0.91, 0.45]

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 7: Compute Loss (Training)                                 │
└─────────────────────────────────────────────────────────────────┘

Predicted scores: [0.87, -0.23, 0.91, 0.45]
True labels:      [1.0,  0.0,   1.0,  1.0]

Loss = MSE(predicted_scores, labels)
     = mean((0.87-1.0)² + (-0.23-0.0)² + (0.91-1.0)² + (0.45-1.0)²)
     = mean(0.0169 + 0.0529 + 0.0081 + 0.3025)
     = 0.095

Backward pass → Update weights

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 8: Compute Metrics (Evaluation)                            │
└─────────────────────────────────────────────────────────────────┘

Predicted scores: [0.87, -0.23, 0.91, 0.45]
                  ↓ Apply sigmoid
Probabilities:    [0.70, 0.44, 0.71, 0.61]
                  ↓ Threshold at 0.5
Predictions:      [1,    0,    1,    1]
True labels:      [1,    0,    1,    1]

Metrics:
  Accuracy:  4/4 = 1.00  ✅
  Precision: 1.00
  Recall:    1.00
  F1:        1.00
  AUC:       1.00

                            ↓

┌─────────────────────────────────────────────────────────────────┐
│ STEP 9: Save Model                                              │
└─────────────────────────────────────────────────────────────────┘

Save to: ./feedback_qa_reward_model/final_model/

Files:
  - config.json         (num_labels: 1, model_type: llama)
  - model.safetensors   (model weights)
  - tokenizer.json      (tokenizer vocabulary)
  - tokenizer_config.json
  - special_tokens_map.json
```

---

## 🔄 Comparison: Old vs New Architecture

### OLD (Binary Classification) ❌

```
Input: [batch, seq_len] → Transformer → Pool (mean/cls) → Linear(hidden → 2)
                                                              ↓
                                                    Output: [batch, 2]
                                                            ↓
                                                    Logits for class 0 and 1
                                                            ↓
                                                    Softmax → [p(0), p(1)]
```

**Problem**: verl doesn't know how to use 2-class outputs!

### NEW (Token Classification) ✅

```
Input: [batch, seq_len] → Transformer → Linear(hidden → 1) for each token
                                                              ↓
                                                    Output: [batch, seq_len, 1]
                                                            ↓
                                                    Take EOS token: [batch, 1]
                                                            ↓
                                                    Sigmoid → probability
```

**Advantage**: verl can extract the EOS token score as the reward!

---

## 🎯 Key Insight: Why EOS Token?

The model learns to encode the **overall quality judgment** in the **EOS (End of Sequence) token**.

```
User: How do I get help finding a job?
Assistant: In this rapidly changing jobs market...

Tokens:    [user] [question...] [asst] [answer...] [EOS]
Scores:    [0.1]  [0.15...]    [0.2]  [0.4...]    [0.87] ← This is the reward!
                                                      ↑
                                        Final quality score
```

Why?
- EOS token sees the **entire sequence** (question + answer)
- It has full context to make a quality judgment
- It's the last token before generation stops
- Natural place to encode a summary score

---

## 📈 Training Loop Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│ Epoch 1/3                                                        │
└─────────────────────────────────────────────────────────────────┘

Batch 1: Load 4 examples → Format → Tokenize → Forward pass
         ↓
         Loss = 0.245
         ↓
         Backward → Update weights

Batch 2: Load 4 examples → Forward → Loss = 0.238 → Update
Batch 3: Load 4 examples → Forward → Loss = 0.251 → Update
...
Batch N: Done

Evaluation:
  Validation accuracy: 0.785
  Validation F1: 0.812

┌─────────────────────────────────────────────────────────────────┐
│ Epoch 2/3                                                        │
└─────────────────────────────────────────────────────────────────┘

Batch 1-N: Train... Loss decreasing... ✅
Evaluation:
  Validation accuracy: 0.845  (improved!)
  Validation F1: 0.867

┌─────────────────────────────────────────────────────────────────┐
│ Epoch 3/3                                                        │
└─────────────────────────────────────────────────────────────────┘

Batch 1-N: Train... Loss = 0.152 ✅
Evaluation:
  Validation accuracy: 0.872  ✅
  Validation F1: 0.891  ✅
  
Model saved! ✅
```

---

## 🔍 Example: Single Sequence Through Pipeline

### Input

```json
{
  "question": "What are common interview mistakes?",
  "section_content": "Being late, not researching the company.",
  "label": 0
}
```

### After Formatting

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

What are common interview mistakes?<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Being late, not researching the company.<|eot_id|>
```

### After Tokenization

```
Token IDs: [128000, 128006, 882, ..., 128009]  (87 tokens)
Label: 0.0  (bad answer)
```

### Model Output

```
Scores for each token:
[0.05, 0.06, 0.08, ..., -0.42]
                          ↑
                    EOS token score: -0.42
```

### Prediction

```
-0.42 → sigmoid → 0.40 → threshold → 0 (bad answer) ✅ Correct!
```

---

## 💡 Summary

1. **Input**: JSON with Q&A pairs and binary labels
2. **Format**: Apply chat template (User: Q, Assistant: A)
3. **Tokenize**: Convert to token IDs, labels to float
4. **Model**: Token classification outputs score per token
5. **Extract**: Take EOS token score as quality judgment
6. **Train**: MSE loss between EOS score and true label
7. **Evaluate**: Sigmoid + threshold for binary metrics
8. **Save**: verl-compatible model with `num_labels=1`

**Key**: The EOS token learns to encode the overall quality of the Q&A pair! 🎯

