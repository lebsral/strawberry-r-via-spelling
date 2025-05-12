import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset

# Config
DATA_FILE = "data/processed/train_spelling.json"  # Change as needed
MODEL_NAME = "Qwen/Qwen3-4B"  # Use Qwen3-4B for compatibility
OUTPUT_DIR = "results/sample_finetune"

# Load Alpaca-format data
with open(DATA_FILE) as f:
    data = json.load(f)
    if isinstance(data, dict) and "examples" in data:
        examples = data["examples"]
    elif isinstance(data, list):
        examples = data
    else:
        raise ValueError(f"Unrecognized data format in {DATA_FILE}")

# Prepare Hugging Face Dataset
# For Alpaca format: concatenate instruction + input as prompt, output as label
prompts = [ex["instruction"] + " " + ex["input"] for ex in examples]
labels = [ex["output"] for ex in examples]

hf_dataset = Dataset.from_dict({"text": prompts, "labels": labels})

# Tokenizer/model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Set pad_token if not present (required for padding)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

def tokenize(batch):
    # For demonstration, use text as input and labels as output
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

tokenized = hf_dataset.map(tokenize, batched=True)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args (very short run for validation)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    save_steps=5,
    save_total_limit=1,
    logging_steps=1,
    report_to=[],
    no_cuda=True,  # Force CPU for local validation
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

print("Starting sample fine-tuning run (local/CPU, 1 epoch)...")
trainer.train()
print("Sample fine-tuning complete. Check logs and output in results/sample_finetune/")

# ---
# For cloud/Unsloth usage:
# - Replace MODEL_NAME with Unsloth-compatible model if needed
# - Remove 'no_cuda=True' and set up GPU environment
# - Adjust batch size and epochs as needed
# - See README and docs/cloud_workflow.md for details
