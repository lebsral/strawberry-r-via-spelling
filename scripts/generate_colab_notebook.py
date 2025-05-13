import os
import sys
import json
import nbformat as nbf
from pathlib import Path

# Add project root to path for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_training_notebook():
    # Create a new notebook
    nb = nbf.v4.new_notebook()

    # Title and introduction
    nb.cells.append(nbf.v4.new_markdown_cell("""# Tokenizer Model Training

This notebook trains the tokenizer model using the Alpaca format. Follow the steps below to:
1. Set up the environment
2. Download the training data
3. Train the model
4. Save the results

## Initial Setup"""))

    # Setup cell - install dependencies
    setup_code = '''%%capture
!pip install transformers torch wandb datasets tqdm
!pip install -q git+https://github.com/huggingface/transformers.git'''
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # Clone repository
    nb.cells.append(nbf.v4.new_markdown_cell("""## Clone Repository and Setup Data"""))
    clone_code = '''!git clone https://github.com/lebsral/raspberry.git
%cd raspberry
!pip install -r requirements.txt'''
    nb.cells.append(nbf.v4.new_code_cell(clone_code))

    # Mount Google Drive
    nb.cells.append(nbf.v4.new_markdown_cell("""## Mount Google Drive
Mount Google Drive to save checkpoints and load data if needed:"""))
    drive_code = '''from google.colab import drive
drive.mount('/content/drive')'''
    nb.cells.append(nbf.v4.new_code_cell(drive_code))

    # Import dependencies
    nb.cells.append(nbf.v4.new_markdown_cell("""## Import Dependencies"""))
    imports = '''import os
import sys
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import wandb'''
    nb.cells.append(nbf.v4.new_code_cell(imports))

    # Configuration
    nb.cells.append(nbf.v4.new_markdown_cell("""## Configuration
Set up training parameters:"""))
    config = '''# Training configuration
config = {
    "model_name": "gpt2",  # Base model to fine-tune
    "train_file": "data/processed/alpaca_examples.json",
    "output_dir": "/content/drive/MyDrive/tokenizer_checkpoints",
    "num_epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "max_length": 512,
}

# Create output directory
os.makedirs(config["output_dir"], exist_ok=True)'''
    nb.cells.append(nbf.v4.new_code_cell(config))

    # Training code
    nb.cells.append(nbf.v4.new_markdown_cell("""## Training Code
Import the training implementation:"""))
    train_code = '''from src.training.train import AlpacaDataset, train

# Initialize wandb
wandb.login()

# Start training
train(
    model_name=config["model_name"],
    train_file=config["train_file"],
    output_dir=config["output_dir"],
    num_epochs=config["num_epochs"],
    batch_size=config["batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    learning_rate=config["learning_rate"],
    max_length=config["max_length"],
)'''
    nb.cells.append(nbf.v4.new_code_cell(train_code))

    # Save results
    nb.cells.append(nbf.v4.new_markdown_cell("""## Save Results
The model checkpoints are automatically saved to Google Drive. You can also download them locally:"""))
    save_code = '''# Download final checkpoint if needed
!zip -r /content/model_checkpoint.zip {config["output_dir"]}
from google.colab import files
files.download("/content/model_checkpoint.zip")'''
    nb.cells.append(nbf.v4.new_code_cell(save_code))

    # Save the notebook
    notebook_path = Path("notebooks/train_model.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    create_training_notebook()
