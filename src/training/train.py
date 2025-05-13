import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlpacaDataset(Dataset):
    """Dataset for Alpaca-format training data"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load and process the data
        with open(data_path, 'r') as f:
            self.examples = json.load(f)

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        # Format as Alpaca instruction format
        prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""

        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Convert to expected format
        return {
            "input_ids": encoded["input_ids"][0],
            "attention_mask": encoded["attention_mask"][0],
            "labels": encoded["input_ids"][0].clone(),  # For causal LM training
        }

def train(
    model_name: str,
    train_file: str,
    val_file: Optional[str] = None,
    output_dir: str = "checkpoints",
    num_epochs: int = 3,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_length: int = 512,
    seed: int = 42,
    use_wandb: bool = True,
):
    """Main training function

    Args:
        model_name: Name/path of pretrained model
        train_file: Path to training data JSON file
        val_file: Optional path to validation data JSON file
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        gradient_accumulation_steps: Number of steps to accumulate gradients
        learning_rate: Learning rate
        max_length: Maximum sequence length
        seed: Random seed
        use_wandb: Whether to use Weights & Biases logging
    """
    # Set seed for reproducibility
    set_seed(seed)

    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project="tokenizer-training")

    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Prepare datasets
    logger.info("Preparing datasets")
    train_dataset = AlpacaDataset(train_file, tokenizer, max_length)
    val_dataset = AlpacaDataset(val_file, tokenizer, max_length) if val_file else None

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        evaluation_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=2,
        report_to="wandb" if use_wandb else "none",
        fp16=True,  # Use mixed precision training
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train
    logger.info("Starting training")
    trainer.train()

    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train tokenizer model")
    parser.add_argument("--model-name", required=True, help="Name/path of pretrained model")
    parser.add_argument("--train-file", required=True, help="Path to training data JSON file")
    parser.add_argument("--val-file", help="Path to validation data JSON file")
    parser.add_argument("--output-dir", default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable Weights & Biases logging")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )
