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
!pip install torch transformers datasets wandb matplotlib seaborn pandas ipywidgets'''
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # Repository setup
    nb.cells.append(nbf.v4.new_markdown_cell("""## Clone Repository and Setup
This cell handles repository setup and configuration."""))

    setup_code = '''# Setup script for Google Colab
import os
import sys
from google.colab import drive

def setup_repository():
    # 1. Start in content directory
    os.chdir('/content')
    print("Starting in directory:", os.getcwd())

    # 2. Define repository and paths
    REPO_NAME = "raspberry"
    REPO_URL = f"https://github.com/lebsral/{REPO_NAME}.git"
    REPO_PATH = os.path.join('/content', REPO_NAME)

    try:
        # 3. Clean up any existing repo directory
        if os.path.exists(REPO_PATH):
            print(f"Found existing repository at {REPO_PATH}")
            !rm -rf $REPO_PATH
            print("✅ Cleaned up existing repository")

        # 4. Clone fresh repository
        print("Cloning fresh repository...")
        !git clone $REPO_URL

        # 5. Change to repository directory
        os.chdir(REPO_PATH)
        print(f"✅ Changed working directory to: {os.getcwd()}")

        # 6. Add repository root to Python path
        if REPO_PATH not in sys.path:
            sys.path.insert(0, REPO_PATH)
            print(f"✅ Added {REPO_PATH} to Python path")

        # 7. Verify setup
        print("\\nVerification:")
        print("-------------")
        print("Repository contents:", os.listdir('.'))
        print("Current working directory:", os.getcwd())
        print("Python path:", REPO_PATH)

        # 8. Test import
        try:
            import src
            print("✅ src module can be imported successfully")
        except ImportError as e:
            print(f"❌ Error importing src module: {e}")

        print("\\n✅ Setup completed successfully!")

    except Exception as e:
        print(f"\\n❌ Error during setup: {e}")
        return False

    return True

# Run the setup
setup_repository()'''
    nb.cells.append(nbf.v4.new_code_cell(setup_code))

    # Mount Google Drive
    nb.cells.append(nbf.v4.new_markdown_cell("""## Mount Google Drive
Mount Google Drive to save checkpoints and load data if needed:"""))
    drive_code = '''from google.colab import drive
drive.mount('/content/drive')'''
    nb.cells.append(nbf.v4.new_code_cell(drive_code))

    # Data Validation
    nb.cells.append(nbf.v4.new_markdown_cell("""## Validate Alpaca-format Data
Check all JSON files in `data/processed/` for schema compliance."""))
    validate_code = '''from src.data.validate_alpaca_schema import AlpacaSchemaValidator
from pathlib import Path

data_dir = Path('data/processed')
english_tokens_path = data_dir / 'english_tokens.json'
validator = AlpacaSchemaValidator(english_tokens_path if english_tokens_path.exists() else None)
reports = validator.validate_dir(data_dir)
for report in reports:
    print(f'\\nFile: {report["file"]}')
    if 'total' in report:
        print(f'  Total examples: {report["total"]}')
        print(f'  Valid: {report["valid"]}')
        print(f'  Invalid: {report["invalid"]}')
        if report['invalid'] > 0:
            for err in report['errors']:
                print(f'    Example #{err["index"]}: {err["errors"]}')
    else:
        print(f'  Error: {report.get("error", "Unknown error")}')'''
    nb.cells.append(nbf.v4.new_code_cell(validate_code))

    # Data Loading
    nb.cells.append(nbf.v4.new_markdown_cell("""## Load and Inspect Training Data
Load Alpaca-format data and print stats and a sample batch."""))
    load_code = '''from src.data.data_loader import TemplateDataLoader, BatchConfig

loader = TemplateDataLoader(
    data_dir=data_dir,
    batch_config=BatchConfig(batch_size=4, max_length=128, shuffle=False)
)
stats = loader.get_stats()
print('Stats:', stats)
for batch in loader.train_batches():
    print('Batch:', batch)
    break  # Show only the first batch'''
    nb.cells.append(nbf.v4.new_code_cell(load_code))

    # Training Configuration
    nb.cells.append(nbf.v4.new_markdown_cell("""## Training Configuration
Set up model training parameters:"""))
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

    # Training
    nb.cells.append(nbf.v4.new_markdown_cell("""## Model Training
Initialize and start the training process:"""))
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
