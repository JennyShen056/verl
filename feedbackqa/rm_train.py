import argparse
import gc
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForTokenClassification,
                          AutoTokenizer, DataCollatorWithPadding, Trainer,
                          TrainerCallback, TrainingArguments)

# Environment setup
for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(k, None)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RewardModelTrainer(Trainer):
    """Custom Trainer with Binary Cross-Entropy loss for reward model training"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss function using Binary Cross-Entropy for verl-compatible reward model.
        
        Model outputs: [batch_size, seq_len, 1] logits
        We take the last token (EOS) logit and apply BCE loss.
        
        Args:
            model: The model being trained
            inputs: Input batch
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        """
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: [batch_size, seq_len, 1]
        
        # Extract EOS token logits (last token in sequence)
        # Shape: [batch_size, seq_len, 1] -> [batch_size, 1] -> [batch_size]
        eos_logits = logits[:, -1, 0]  # Last token, first (only) class
        
        # Apply sigmoid and compute BCE loss
        # BCE expects predictions and targets in [0, 1]
        loss = F.binary_cross_entropy_with_logits(
            eos_logits, 
            labels.float(),
            reduction='mean'
        )
        
        return (loss, outputs) if return_outputs else loss


class TrainingAccuracyCallback(TrainerCallback):
    """Log training accuracy on each eval without recursive callbacks"""

    def __init__(self):
        self.trainer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not hasattr(self, "trainer") or self.trainer is None:
            logger.warning("TrainingAccuracyCallback has no trainer reference attached")

    def on_evaluate(self, args, state, control, **kwargs):
        try:
            if self.trainer is None or not hasattr(self.trainer, "train_dataset"):
                return control

            ds = self.trainer.train_dataset
            n = min(1000, len(ds))
            # deterministic subset tied to seed
            rng = np.random.default_rng(42 + state.global_step)
            indices = rng.choice(len(ds), size=n, replace=False)
            train_subset = ds.select(indices.tolist())

            # predict does not trigger on_evaluate again
            pred_output = self.trainer.predict(train_subset, metric_key_prefix="train")
            metrics = pred_output.metrics  # keys like train_loss, train_accuracy, etc.

            # Log through Trainer so report_to="wandb" handles forwarding
            self.trainer.log(metrics)

        except Exception as e:
            logger.exception("Failed to log training metrics on eval")
        return control


class ClassificationDataLoader:
    """Load pre-formatted binary classification data for feedback QA reward model training"""

    def __init__(self, tokenizer=None):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer

    def format_with_chat_template(self, question: str, section_content: str) -> str:
        """Format question and answer using chat template"""
        if self.tokenizer is None:
            # Fallback to simple format if no tokenizer
            return f"Question: {question}\n\nAnswer: {section_content}"
        
        # Use chat template if available
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": section_content}
        ]
        
        try:
            # Try to use apply_chat_template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
            else:
                # Fallback format
                text = f"Question: {question}\n\nAnswer: {section_content}"
        except Exception as e:
            self.logger.warning(f"Chat template failed, using fallback format: {e}")
            text = f"Question: {question}\n\nAnswer: {section_content}"
        
        return text

    def load_feedback_data(
        self,
        train_file: str,
        valid_file: str,
        test_file: str,
    ) -> Dict[str, List[Dict]]:
        """Load feedback QA data from train/valid/test files"""
        data = {"train": [], "valid": [], "test": []}
        
        # Load training data
        if os.path.exists(train_file):
            self.logger.info(f"Loading training data from: {train_file}")
            with open(train_file, "r") as f:
                train_raw = json.load(f)
            
            for item in train_raw:
                text = self.format_with_chat_template(
                    item["question"], 
                    item["section_content"]
                )
                data["train"].append({
                    "text": text,
                    "label": item["label"],
                    "source": "feedback_qa"
                })
            self.logger.info(f"Loaded {len(data['train'])} training examples")
        
        # Load validation data
        if os.path.exists(valid_file):
            self.logger.info(f"Loading validation data from: {valid_file}")
            with open(valid_file, "r") as f:
                valid_raw = json.load(f)
            
            for item in valid_raw:
                text = self.format_with_chat_template(
                    item["question"], 
                    item["section_content"]
                )
                data["valid"].append({
                    "text": text,
                    "label": item["label"],
                    "source": "feedback_qa"
                })
            self.logger.info(f"Loaded {len(data['valid'])} validation examples")
        
        # Load test data
        if os.path.exists(test_file):
            self.logger.info(f"Loading test data from: {test_file}")
            with open(test_file, "r") as f:
                test_raw = json.load(f)
            
            for item in test_raw:
                text = self.format_with_chat_template(
                    item["question"], 
                    item["section_content"]
                )
                data["test"].append({
                    "text": text,
                    "label": item["label"],
                    "source": "feedback_qa"
                })
            self.logger.info(f"Loaded {len(data['test'])} test examples")
        
        return data


class BinaryClassificationRewardModelTrainer:
    """Main trainer for feedback QA reward model with binary classification"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        output_dir: str = "./feedback_qa_reward_model",
        max_length: int = 1024,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_length = max_length
        self.seed = seed

        self.tokenizer = None
        self.model = None

        set_seed(seed)
        self.logger = logging.getLogger(__name__)

    def load_formatted_datasets(
        self,
        train_file: Optional[str] = None,
        valid_file: Optional[str] = None,
        test_file: Optional[str] = None,
    ) -> DatasetDict:
        """Load and process formatted datasets for binary classification"""
        # Initialize loader with tokenizer for chat template
        loader = ClassificationDataLoader(tokenizer=self.tokenizer)

        # Load pre-split data
        data = loader.load_feedback_data(
            train_file=train_file,
            valid_file=valid_file,
            test_file=test_file,
        )

        if not data["train"]:
            raise ValueError("No training data loaded! Check your file paths.")

        # Process each split
        for split_name in ["train", "valid", "test"]:
            examples = data[split_name]
            if not examples:
                self.logger.warning(f"No {split_name} data loaded!")
                continue

            # Validate labels
            label_distribution = {"0": 0, "1": 0}
            for i, example in enumerate(examples):
                label = example["label"]
                if label not in [0, 1, 0.0, 1.0]:
                    self.logger.warning(
                        f"Non-binary label found in {split_name} at index {i}: {label}. Converting to binary."
                    )
                    example["label"] = 1 if float(label) > 0.5 else 0
                
                label_distribution[str(int(example["label"]))] += 1

            # Log statistics
            total = len(examples)
            self.logger.info(f"{split_name.capitalize()} set: {total} examples")
            self.logger.info(f"  Label distribution: {label_distribution}")
            if total > 0:
                pos_pct = (label_distribution['1'] / total) * 100
                self.logger.info(f"  Positive class: {pos_pct:.1f}%")

        # Create datasets from the pre-split data
        train_texts = [ex["text"] for ex in data["train"]]
        train_labels = [int(ex["label"]) for ex in data["train"]]
        
        val_texts = [ex["text"] for ex in data["valid"]]
        val_labels = [int(ex["label"]) for ex in data["valid"]]
        
        test_texts = [ex["text"] for ex in data["test"]]
        test_labels = [int(ex["label"]) for ex in data["test"]]

        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

        # Create DatasetDict
        dataset_dict = DatasetDict(
            {"train": train_dataset, "validation": val_dataset, "test": test_dataset}
        )

        return dataset_dict

    def setup_model_and_tokenizer(self):
        """Setup model and tokenizer for binary classification"""
        self.logger.info(f"Loading tokenizer and model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.pad_token_id = (
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        )
        self.tokenizer.padding_side = "right"

        # Load model config and set for verl-compatible token classification
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        config.num_labels = 1  # Single scalar reward per token (verl-compatible)
        config.problem_type = "single_label_classification"

        # Ensure proper classification head initialization
        config.classifier_dropout = 0.1  # Add dropout for regularization
        config.pad_token_id = self.tokenizer.pad_token_id  # important

        # Load model - using TokenClassification for verl compatibility
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # keep model and generation configs in sync
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if (
            hasattr(self.model, "generation_config")
            and self.model.generation_config is not None
        ):
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        # gradient checkpointing already forces use_cache False, this is safe to set too
        self.model.config.use_cache = False

        self.logger.info("Model and tokenizer setup complete")
        self.logger.info(
            f"Model configuration: {config.num_labels} labels, {config.problem_type}"
        )
        self.logger.info(
            f"Classification head dropout: {getattr(config, 'classifier_dropout', 'default')}"
        )
        self.logger.info("Loss function: Binary Cross-Entropy (BCE) - optimal for binary labels")

    def tokenize_function(self, examples):
        """Tokenize text and convert binary labels to float scores for verl compatibility"""
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None,
        )
        # Convert binary labels (0, 1) to float scores for token classification
        # This makes it verl-compatible (single scalar reward)
        tokenized["labels"] = [float(label) for label in examples["label"]]
        return tokenized

    def compute_metrics(self, eval_pred):
        """Comprehensive binary classification metrics for verl-compatible model"""
        predictions, labels = eval_pred

        # For verl-compatible model (num_labels=1), predictions are shape [batch_size, seq_len, 1]
        # We take the last token's prediction (EOS token) and apply sigmoid
        # Reshape: [batch_size, seq_len, 1] -> take last token -> [batch_size, 1] -> [batch_size]
        if len(predictions.shape) == 3:
            # Token classification output: [batch_size, seq_len, 1]
            probs = torch.sigmoid(torch.from_numpy(predictions[:, -1, 0])).numpy()
        else:
            # Fallback for shape [batch_size, 1]
            probs = torch.sigmoid(torch.from_numpy(predictions[:, 0])).numpy()
        
        predicted_classes = (probs > 0.5).astype(int)
        labels = labels.astype(int)

        # Calculate comprehensive metrics
        accuracy = accuracy_score(labels, predicted_classes)
        precision = precision_score(
            labels, predicted_classes, average="binary", zero_division=0
        )
        recall = recall_score(
            labels, predicted_classes, average="binary", zero_division=0
        )
        f1 = f1_score(labels, predicted_classes, average="binary", zero_division=0)

        # ROC-AUC (using probabilities)
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0  # In case of single class in labels

        # Confusion matrix
        cm = confusion_matrix(labels, predicted_classes)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "specificity": specificity,
            "true_positives": int(tp),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
        }

    def train(
        self,
        dataset: DatasetDict,
        learning_rate: float = 1e-5,
        num_epochs: int = 2,
        batch_size: int = 16,
    ):
        """Train the unified reward model with binary classification"""
        self.logger.info(
            "Starting unified reward model training (binary classification)..."
        )

        tokenized_datasets = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
        )

        # Calculate dynamic training schedule
        total_steps = (len(tokenized_datasets["train"]) // batch_size) * num_epochs
        calculated_warmup_steps = int(0.1 * total_steps)
        calculated_eval_steps = max(200, int(0.15 * total_steps))

        self.logger.info(f"Training Schedule:")
        self.logger.info(f"  Total steps: {total_steps}")
        self.logger.info(f"  Warmup steps: {calculated_warmup_steps} (10%)")
        self.logger.info(
            f"  Eval steps: {calculated_eval_steps} (~{total_steps//calculated_eval_steps} evaluations)"
        )
        self.logger.info(f"  Logging steps: {50}")

        # Training arguments optimized for binary classification
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=calculated_warmup_steps,
            weight_decay=0.01,
            learning_rate=learning_rate,
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=20,  # Frequent logging for monitoring
            eval_strategy="steps",
            eval_steps=calculated_eval_steps,
            save_strategy="steps",
            save_steps=calculated_eval_steps,  # Align with eval_steps
            load_best_model_at_end=True,
            metric_for_best_model="f1",  # Use F1 score for best model selection
            greater_is_better=True,
            report_to="wandb",
            run_name=f"unified_rm_classification_{self.model_name.split('/')[-1]}",
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            fp16=False,
            bf16=True,
            remove_unused_columns=False,
            # Additional classification-specific settings
            label_smoothing_factor=0.0,  # No label smoothing for binary classification
            seed=self.seed,
        )

        # Data collator
        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, padding=True, return_tensors="pt"
        )

        # Create training accuracy callback
        training_callback = TrainingAccuracyCallback()
        # Use custom trainer with BCE loss for binary classification
        trainer = RewardModelTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[training_callback],
        )

        # give the callback a handle to the trainer
        training_callback.trainer = trainer

        # Train
        self.logger.info("Starting training...")
        trainer.train()

        # Evaluate on all splits
        self.logger.info("Evaluating on all splits...")

        train_results = trainer.evaluate(eval_dataset=tokenized_datasets["train"])
        val_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
        test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])

        self.logger.info(f"Train results: {train_results}")
        self.logger.info(f"Validation results: {val_results}")
        self.logger.info(f"Test results: {test_results}")

        # Save final model
        self.logger.info("Saving final model...")
        trainer.save_model(f"{self.output_dir}/final_model")
        self.tokenizer.save_pretrained(f"{self.output_dir}/final_model")

        # Save comprehensive training summary
        summary = {
            "model_name": self.model_name,
            "training_args": training_args.to_dict(),
            "classification_type": "binary",
            "task": "feedback_qa_reward_model",
            "results": {
                "train": train_results,
                "validation": val_results,
                "test": test_results,
            },
            "dataset_sizes": {
                "train": len(tokenized_datasets["train"]),
                "validation": len(tokenized_datasets["validation"]),
                "test": len(tokenized_datasets["test"]),
            },
            "model_config": {
                "num_labels": 1,
                "problem_type": "single_label_classification",
                "max_length": self.max_length,
                "output_type": "single_scalar_reward_per_token",
                "loss_function": "binary_cross_entropy",
                "verl_compatible": True,
            },
        }

        with open(f"{self.output_dir}/training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info("Training completed!")
        self.logger.info(
            f"Final Test Accuracy: {test_results.get('eval_accuracy', 0):.4f}"
        )
        self.logger.info(f"Final Test F1: {test_results.get('eval_f1', 0):.4f}")

        return trainer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Feedback QA Reward Model Trainer (Binary Classification)"
    )

    # Data loading arguments
    parser.add_argument(
        "--train_file",
        type=str,
        default="feedback_train_rm.json",
        help="Path to training data file",
    )
    parser.add_argument(
        "--valid_file",
        type=str,
        default="feedback_valid_rm.json",
        help="Path to validation data file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="feedback_test_rm.json",
        help="Path to test data file",
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Pre-trained model name or path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./feedback_qa_reward_model",
        help="Output directory for model and logs",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length"
    )

    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_arguments()

    # Validate arguments
    if not all([args.train_file, args.valid_file, args.test_file]):
        raise ValueError("All three data files (train, valid, test) must be specified")

    # Check if files exist
    for file_path in [args.train_file, args.valid_file, args.test_file]:
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")

    # Initialize trainer
    trainer = BinaryClassificationRewardModelTrainer(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_length=args.max_length,
        seed=args.seed,
    )

    # Setup model and tokenizer first (needed for chat template)
    trainer.setup_model_and_tokenizer()

    # Load datasets (tokenizer is now available for chat template)
    dataset = trainer.load_formatted_datasets(
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
    )

    # Initialize wandb
    os.makedirs(args.output_dir, exist_ok=True)
    wandb.init(
        project="feedback_qa_reward_model",
        name=f"feedback_qa_rm_{args.model_name.split('/')[-1]}",
        config=vars(args),
    )

    # Train
    trainer.train(
        dataset,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
    )

    # Cleanup
    wandb.finish()
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
