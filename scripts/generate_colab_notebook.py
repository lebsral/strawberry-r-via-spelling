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

    # GPU Check
    gpu_check = '''# Check if we have GPU access
import torch

def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No GPU found! This notebook requires a GPU runtime. "
            "Go to Runtime > Change runtime type and select GPU."
        )
    print("✅ GPU is available:", torch.cuda.get_device_name(0))
    print("   CUDA Version:", torch.version.cuda)

check_gpu()'''
    nb.cells.append(nbf.v4.new_code_cell(gpu_check))

    # Setup cell - install dependencies with progress
    setup_code = '''%%capture --no-stderr
import sys
from IPython.display import clear_output

def install_with_progress(packages):
    from IPython.display import display, HTML
    import subprocess
    import sys

    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        clear_output(wait=True)
    print("✅ All packages installed successfully!")

# First uninstall any existing torch and transformers
print("Removing existing installations...")
subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio", "transformers"])

# Install specific versions known to work together
base_packages = [
    "torch==2.1.2",
    "torchvision==0.16.2",
    "torchaudio==2.1.2",
    "transformers==4.37.2",
    "accelerate==0.27.2",
    "datasets",
    "wandb",
    "matplotlib",
    "seaborn",
    "pandas",
    "ipywidgets",
    "tqdm"
]

# GPU-specific packages
gpu_packages = [
    "unsloth",
    "xformers==0.0.23.post1",
    "flash-attn==2.3.6"
]

print("Installing base packages...")
install_with_progress(base_packages)

print("\\nInstalling GPU-specific packages...")
install_with_progress(gpu_packages)'''
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

    # Model and data loading
    model_setup = '''import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel
import warnings
warnings.filterwarnings('ignore')

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
    if "only works on NVIDIA GPUs" in str(e):
        print("\\nThis error indicates you're not using an NVIDIA GPU.")
        print("Please ensure you're using a GPU runtime and it's properly initialized.")
        print("Go to Runtime > Change runtime type and select GPU.")
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
