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
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl==0.15.2 triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
    !pip install --no-deps unsloth

# Install other dependencies
!pip install torch==2.1.2 transformers==4.37.2 wandb matplotlib seaborn pandas ipywidgets tqdm --quiet

print("\\nVerifying installations:")
import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")'''
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # Repository setup
    repo_setup = '''# Clone and set up repository
import os
import sys
from pathlib import Path

# Define repository information
REPO_URL = "https://github.com/lebsral/strawberry-r-via-spelling.git"
REPO_NAME = "strawberry-r-via-spelling"
BASE_DIR = "/content"

print(f"Starting in directory: {BASE_DIR}")

# Clean up any existing repository
repo_path = os.path.join(BASE_DIR, REPO_NAME)
if os.path.exists(repo_path):
    print(f"Cleaning up existing repository...")
    !rm -rf $repo_path

# Clone fresh repository
print("Cloning repository...")
!git clone $REPO_URL

# Change to repository directory
%cd $repo_path
print(f"Changed to: {os.getcwd()}")

# Add to Python path
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)
    print(f"Added {repo_path} to Python path")

print("\\nVerifying setup:")
!ls -la
print(f"Current directory: {os.getcwd()}")

try:
    import src
    print("✅ src module imported successfully")
except ImportError as e:
    print(f"❌ Error importing src module: {e}")
    raise'''
    nb.cells.append(nbf.v4.new_code_cell(repo_setup))

    # Model and data loading
    model_setup = '''import torch
import warnings
warnings.filterwarnings('ignore')

# Check for GPU availability first
if not torch.cuda.is_available():
    raise RuntimeError("This notebook requires a GPU runtime. Go to Runtime > Change runtime type and select GPU.")

# Import unsloth first to ensure optimizations are applied
from unsloth import FastLanguageModel
from transformers.models.auto import AutoTokenizer

# Load Qwen3-4B model and tokenizer
model_name = "Qwen/Qwen1.5-4B"  # Using Qwen3-4B as specified in docs
print("Loading model and tokenizer...")

try:
    # First load the tokenizer separately
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        pad_token="<|endoftext|>"
    )

    # Initialize model with unsloth for faster training
    model, _ = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=512,
        dtype=torch.bfloat16,
        load_in_4bit=True,  # Quantization for memory efficiency
        trust_remote_code=True
    )

    # Disable thinking mode as per project policy
    model.config.enable_thinking = False

    print("✅ Model and tokenizer loaded successfully")

    # Test tokenization
    test_text = "Hello, how are you?"
    tokens = tokenizer(test_text, return_tensors="pt")
    print("\\nTest tokenization:")
    print(f"Input text: {test_text}")
    print(f"Tokenized: {tokens}")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise'''
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
    bf16=True,  # Use bfloat16 for better numerical stability
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    optim="adamw_torch_fused",  # Use fused optimizer for better performance
)

print("✅ Training configuration ready")'''
    nb.cells.append(nbf.v4.new_code_cell(training_setup))

    # Training
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
