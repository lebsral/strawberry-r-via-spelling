# Google Colab Setup Guide

This guide provides detailed instructions for running project notebooks in Google Colab, ensuring proper repository setup, Python path configuration, and environment management.

---

## 1. Initial Setup

### a. Repository Setup

Add this code block at the start of your Colab notebook to clone and set up the repository:

```python
# 1. Start in a clean state
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
print("\nVerification:")
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

print("\n✅ Setup completed successfully!")
```

### b. GPU Runtime Selection

1. Click "Runtime" in the Colab menu
2. Select "Change runtime type"
3. Choose "GPU" as the hardware accelerator
4. Click "Save"

### c. Package Installation

Install required packages (adjust as needed for your notebook):

```python
# Install project requirements
!pip install -r requirements.txt

# Install additional Colab-specific packages
!pip install unsloth xformers tqdm
```

---

## 2. Environment Variables

Set up environment variables for API keys and configuration:

```python
import os
from dotenv import load_dotenv

# Option 1: Load from .env file if you've uploaded it
load_dotenv()

# Option 2: Set variables directly (for testing only)
os.environ["WANDB_API_KEY"] = "your-key-here"  # Replace with your actual key
```

---

## 3. Data Management

### a. Accessing Project Data

The repository's data directory structure will be available after cloning:

```python
from pathlib import Path
import json

# Example: Load component tokens data
data_dir = Path("data/processed")
with open(data_dir / "component_tokens.json") as f:
    component_data = json.load(f)

# Example: Load training examples
with open(data_dir / "train_examples.json") as f:
    train_data = json.load(f)
```

### b. Component Token Analysis

New functionality for analyzing component tokens:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create DataFrame from component tokens
df = pd.DataFrame(component_data['component_tokens'])

# Analyze usage patterns
print("Top 10 Component Tokens by Usage:")
print(df.nlargest(10, 'usage_count')[['token', 'usage_count']])

# Plot distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='usage_count', bins=30)
plt.title('Component Token Usage Distribution')
plt.show()

# Position analysis
position_stats = pd.DataFrame(df['position_stats'].tolist())
print("\nPosition Statistics:")
print(f"Average prefix usage: {position_stats['prefix'].mean():.2f}")
print(f"Average infix usage: {position_stats['infix'].mean():.2f}")
print(f"Average suffix usage: {position_stats['suffix'].mean():.2f}")
```

### c. Uploading Additional Data

For data not in the repository:

```python
from google.colab import files

# Upload files interactively
uploaded = files.upload()

# Or mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

---

## 4. Best Practices

### a. Session Management

- Save important outputs frequently
- Keep code modular and checkpointed
- Use `@retry` decorators for network operations
- Handle session reconnections gracefully

### b. Resource Monitoring

```python
# Monitor GPU usage
!nvidia-smi

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
```

### c. Error Handling

```python
import logging
logging.basicConfig(level=logging.INFO)

try:
    # Your code here
    pass
except Exception as e:
    logging.error(f"Error occurred: {e}")
    # Handle gracefully
```

---

## 5. Common Issues and Solutions

### a. Import Errors

If you see "module not found" errors:
1. Verify the repository was cloned successfully
2. Check that the Python path includes the repository root
3. Ensure all requirements are installed
4. Try restarting the runtime if imports still fail

### b. GPU Issues

If GPU is not detected or CUDA errors occur:
1. Verify GPU runtime is selected
2. Check GPU availability with `!nvidia-smi`
3. Restart runtime if GPU was recently enabled
4. Reduce batch sizes if out of memory

### c. Data Access Issues

If data files are not found:
1. Verify current working directory
2. Check file paths are relative to repository root
3. Ensure data files were included in git clone
4. Upload missing files manually if needed

---

## 6. Example Notebook Structure

Here's a recommended structure for your Colab notebooks:

1. Repository Setup (use code from Section 1)
2. Environment Configuration
   ```python
   # Import required packages
   import torch
   import transformers
   # ... other imports
   
   # Configure environment
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")
   ```
3. Data Loading and Analysis
   ```python
   # Load component tokens
   with open("data/processed/component_tokens.json") as f:
       component_data = json.load(f)
   
   # Analyze distribution
   df = pd.DataFrame(component_data['component_tokens'])
   print("Token distribution:", df['usage_count'].describe())
   ```
4. Main Processing/Training Code
5. Results and Visualization
6. Cleanup and Resource Release

---

## 7. Available Notebooks

The project includes several Colab notebooks:

1. `colab_alpaca_data_validation_and_loading.ipynb`
   - Validates all datasets
   - Loads and inspects data batches
   - Runs basic evaluations

2. `token_validation.ipynb`
   - Validates token extraction
   - Analyzes token statistics
   - Checks for data quality issues

3. `template_analysis.ipynb`
   - Analyzes template distribution
   - Validates template variations
   - Checks coverage across categories

4. `template_performance.ipynb`
   - Evaluates model performance by template
   - Analyzes error patterns
   - Generates performance reports

---

## 8. References

- [Cloud Workflow Guide](cloud_workflow.md)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Project README](../README.md)

---

**Note:** Always save your notebooks to your Google Drive or GitHub to preserve your work between sessions. 
