import os
import sys
import json
import nbformat as nbf
from pathlib import Path

def create_training_notebook():
    # Create a new notebook
    nb = nbf.v4.new_notebook()

    # Title and introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# Tokenizer Model Training in Colab

This notebook trains the tokenizer model using the Alpaca format. Follow the steps below to:
1. Set up the environment
2. Clone and configure the repository
3. Load and validate data
4. Train the model
5. Save the results

## Initial Setup"""))

    # Setup cell - install dependencies
    setup_code = '''%%capture
# Install base requirements
!pip install torch transformers datasets wandb matplotlib seaborn pandas ipywidgets

# Install cloud-specific packages (required for GPU training)
!pip install unsloth xformers tqdm'''
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # Repository setup
    repo_setup = '''# 1. Start in a clean state
import os
import sys
import shutil
from pathlib import Path

# 2. Define repository and setup information
REPO_URL = "https://github.com/lebsral/strawberry-r-via-spelling.git"
REPO_NAME = "strawberry-r-via-spelling"
BASE_DIR = "/content"

print(f"Starting in directory: {BASE_DIR}")

# 3. Clean up any existing repository
repo_path = os.path.join(BASE_DIR, REPO_NAME)
if os.path.exists(repo_path):
    print(f"Found existing repository at {repo_path}")
    try:
        shutil.rmtree(repo_path)
        print("✅ Cleaned up existing repository")
    except Exception as e:
        print(f"❌ Error cleaning up repository: {e}")
        sys.exit(1)

# 4. Clone fresh repository
print("Cloning fresh repository...")
clone_command = f"git clone {REPO_URL}"
if os.system(clone_command) != 0:
    print("❌ Error cloning repository")
    sys.exit(1)

# 5. Change to repository directory
try:
    os.chdir(repo_path)
    print(f"✅ Changed working directory to: {repo_path}")
except Exception as e:
    print(f"❌ Error changing directory: {e}")
    sys.exit(1)

# 6. Add repository root to Python path
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
    print(f"✅ Added {repo_path} to Python path")

# 7. Verify setup
print("\\nVerification:")
print("-------------")
print(f"Repository contents: {os.listdir('.')}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {repo_path}")

# 8. Test src import
try:
    import src
    print("✅ src module can be imported successfully")
except ImportError as e:
    print(f"❌ Error importing src module: {e}")
    sys.exit(1)

print("\\n✅ Setup completed successfully!")'''
    nb.cells.append(nbf.v4.new_code_cell(repo_setup))

    # GPU Setup
    nb.cells.append(nbf.v4.new_markdown_cell("""## GPU Setup

First, ensure you have selected a GPU runtime:
1. Click 'Runtime' in the menu
2. Select 'Change runtime type'
3. Choose 'GPU' as the hardware accelerator
4. Click 'Save'"""))

    # Model and data loading
    model_setup = '''from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from unsloth import FastLanguageModel

# Load Qwen3-4B model and tokenizer
model_name = "Qwen/Qwen1.5-4B"  # Using Qwen3-4B as specified in docs
print("Loading model and tokenizer...")

# Initialize with unsloth for faster training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=True,  # Quantization for memory efficiency
)

# Disable thinking mode as per project policy
model.config.enable_thinking = False

print("✅ Model and tokenizer loaded successfully")'''
    nb.cells.append(nbf.v4.new_code_cell(model_setup))

    # Data loading
    data_loading = '''# Load and prepare training data
from pathlib import Path
import json

data_dir = Path("data/processed")
print("Loading training data...")

# Load component tokens data
with open(data_dir / "component_tokens.json") as f:
    component_data = json.load(f)

# Load training examples
with open(data_dir / "train_examples.json") as f:
    train_data = json.load(f)

print("✅ Data loaded successfully")'''
    nb.cells.append(nbf.v4.new_code_cell(data_loading))

    # Training setup
    training_setup = '''from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
import wandb

# Initialize wandb
wandb.init(project="tokenizer-training")

# Set up training arguments
training_args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
)

print("✅ Training configuration ready")'''
    nb.cells.append(nbf.v4.new_code_cell(training_setup))

    # Start training
    training = '''# Start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

print("Starting training...")
trainer.train()
print("✅ Training completed")

# Save the model
output_dir = "final_model"
trainer.save_model(output_dir)
print(f"✅ Model saved to {output_dir}")'''
    nb.cells.append(nbf.v4.new_code_cell(training))

    # Save notebook
    notebook_dir = Path("notebooks")
    notebook_dir.mkdir(exist_ok=True)
    notebook_path = notebook_dir / "train_model.ipynb"

    with open(notebook_path, "w") as f:
        nbf.write(nb, f)

    print(f"Notebook created at {notebook_path}")

if __name__ == "__main__":
    create_training_notebook()
