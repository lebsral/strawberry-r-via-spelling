import nbformat as nbf

nb = nbf.v4.new_notebook()

# Create markdown cell with introduction
markdown_intro = '''# Cloud GPU Environment Setup Guide

This notebook provides step-by-step instructions for setting up GPU-accelerated environments for the LLM Spelling Project using:
1. Google Colab
2. Lightning.ai

## Prerequisites
- A Google account (for Colab)
- A Lightning.ai account
- Weights & Biases account
- Hugging Face account

## Environment Setup Overview
1. Install required packages
2. Configure authentication
3. Set up data synchronization
4. Test GPU availability
5. Run example fine-tuning code'''

nb['cells'] = [nbf.v4.new_markdown_cell(markdown_intro)]

# Add code cell for GPU verification
code_gpu = '''# Verify GPU availability
!nvidia-smi'''
nb['cells'].append(nbf.v4.new_code_cell(code_gpu))

# Add installation cell
code_install = '''!pip install unsloth torch transformers datasets wandb dspy lightning matplotlib seaborn pandas jupyter notebook ipywidgets'''
nb['cells'].append(nbf.v4.new_code_cell(code_install))

# Add authentication cell
code_auth = '''# Weights & Biases setup
import wandb
wandb.login()  # You'll need to enter your API key

# Hugging Face setup
from huggingface_hub import login
login()  # You'll need to enter your token'''
nb['cells'].append(nbf.v4.new_code_cell(code_auth))

# Add repository setup cell
code_repo = '''!git clone https://github.com/yourusername/raspberry.git
!cd raspberry'''
nb['cells'].append(nbf.v4.new_code_cell(code_repo))

# Add example fine-tuning code
code_finetune = '''from unsloth import FastLanguageModel
import torch

# Initialize model with Unsloth optimizations
model_name = "meta-llama/Llama-2-7b-hf"  # Example model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Example training configuration
training_config = {
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
}

# The actual training code will depend on your specific dataset and requirements'''
nb['cells'].append(nbf.v4.new_code_cell(code_finetune))

# Add Lightning.ai setup instructions
markdown_lightning = '''## 2. Lightning.ai Setup

### 2.1 Initial Setup
1. Go to [Lightning.ai](https://lightning.ai/lars/home)
2. Create a new project or open existing
3. Select a GPU-enabled compute instance'''
nb['cells'].append(nbf.v4.new_markdown_cell(markdown_lightning))

# Add Lightning app configuration
code_lightning = '''%%writefile lightning.app
name: llm-spelling-project
compute:
  type: gpu
  size: medium  # Adjust based on needs
requirements:
  - unsloth
  - transformers
  - wandb
  - dspy'''
nb['cells'].append(nbf.v4.new_code_cell(code_lightning))

# Add model syncing instructions
markdown_sync = '''## 3. Syncing Results and Models

### 3.1 Using Weights & Biases
W&B is the primary method for tracking experiments and syncing models:'''
nb['cells'].append(nbf.v4.new_markdown_cell(markdown_sync))

# Add W&B code example
code_wandb = '''import wandb

# Initialize a new run
wandb.init(
    project="llm-spelling",
    name="fine-tuning-run-1",
    config=training_config
)

# Log metrics during training
wandb.log({"loss": 0.5, "accuracy": 0.95})

# Save model artifacts
wandb.save("./model.pt")'''
nb['cells'].append(nbf.v4.new_code_cell(code_wandb))

# Add HuggingFace Hub instructions
markdown_hf = '''### 3.2 Using Hugging Face Hub
For sharing models and datasets:'''
nb['cells'].append(nbf.v4.new_markdown_cell(markdown_hf))

# Add HuggingFace code example
code_hf = '''from huggingface_hub import push_to_hub

# Save model to Hugging Face Hub
model.push_to_hub("your-username/model-name")
tokenizer.push_to_hub("your-username/model-name")'''
nb['cells'].append(nbf.v4.new_code_cell(code_hf))

# Add troubleshooting section
markdown_trouble = '''## 4. Troubleshooting Tips

### Common Issues and Solutions

1. **GPU Out of Memory**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use 4-bit or 8-bit quantization
   ```python
   model.enable_gradient_checkpointing()
   ```

2. **Colab Runtime Disconnection**
   - Save checkpoints frequently
   - Use W&B for experiment tracking
   - Keep browser tab active

3. **Package Conflicts**
   - Create a fresh environment
   - Install packages in the correct order
   - Check compatibility matrix

4. **Authentication Issues**
   - Verify API keys in environment variables
   - Check token permissions
   - Ensure proper login sequence

### Best Practices

1. **Regular Checkpointing**
   - Save model state every N steps
   - Use W&B for experiment tracking
   - Keep local copies of important data

2. **Resource Management**
   - Monitor GPU memory usage
   - Use appropriate batch sizes
   - Clean up unused variables

3. **Version Control**
   - Commit code changes regularly
   - Use meaningful commit messages
   - Tag important versions'''
nb['cells'].append(nbf.v4.new_markdown_cell(markdown_trouble))

# Write the notebook to a file
nbf.write(nb, 'cloud_gpu_setup.ipynb')
