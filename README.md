# LLM Spelling Project

## Project Overview

This project explores whether training a language model (LLM) on spelling tasks improves its ability to answer position and count questions about words. The experiment is structured using Taskmaster for rigorous, task-driven development and reproducibility.

## Component Token Processing (New)

The project now features sophisticated handling of multi-word tokens:

- **Intelligent Token Breakdown:** Analyzes compound words (e.g., "overboard", "underestimate") to extract meaningful component tokens
- **Usage-Based Dataset Generation:** Creates examples based on component token frequency
- **Balanced Distribution:** Ensures even coverage across different template categories
- **Key Statistics:**
  - 2,716 unique component tokens identified
  - Most common components: "over" (1090 uses), "less" (921 uses), "out" (633 uses)
  - Training set: 4,122 examples from 1,358 component tokens
  - Validation/Test sets: 2,716 examples each from 679 tokens

For detailed information about the component token extraction process, see [docs/token_extraction.md](docs/token_extraction.md#component-token-extraction-new).

## Qwen3-4B Migration

- The project uses the Qwen3-4B model exclusively, always in non-thinking mode (`enable_thinking=False`). Thinking mode is strictly prohibited and enforced in code. Any attempt to use thinking mode will raise an error.
- **Tokenizer-only workflows are the default for all data preparation and token extraction tasks.**
- **The English token extraction script now outputs only `data/processed/english_tokens.json` (as a dict with a `tokens` key). The legacy `.txt` output has been removed. All downstream code and analysis should use the `.json` file.**
- The full model is only loaded for inference or evaluation, not for data prep.
- Scripts and imports are enforced to follow the `src/` layout, and a Cursor rule prevents import errors (see `.cursor/rules/module_imports.mdc`).
- Only the English token subset (~50k tokens, see `english_tokens.json`) is used for all experiments.
- **Tokenizer compatibility is strictly enforced and audited.**
  - See [docs/templates.md](docs/templates.md#qwen3-4b-tokenizer-compatibility-audit-2024-06-12), [docs/data_format.md](docs/data_format.md#qwen3-4b-tokenizer-compatibility-and-data-validation-2024-06-12), and [docs/token_extraction.md](docs/token_extraction.md#qwen3-4b-tokenizer-compatibility-audit-2024-06-12) for details on the audit, validation process, and requirements.
- Model configuration uses specific sampling parameters: Temperature=0.6, TopP=0.95, TopK=20, MinP=0 (for non-thinking mode).
- All scripts, data, and evaluation are Qwen3-4B-specific.
- **Task 15 (Qwen3-4B compatibility migration) is complete. All code, data, and documentation have been audited and updated to enforce non-thinking mode. All subtasks are done or cancelled as appropriate.**
- **CI/CD safeguards for thinking mode are not implemented and not planned. Enforcement is via code and documentation only.**
- **Project policy audit (2024-06-11):** The entire codebase, documentation, and configuration were audited. All code, configs, and docs strictly enforce non-thinking mode. No references to thinking mode remain except as explicit prohibitions. See `/docs/analysis.md`, `/docs/templates.md`, `/docs/data_format.md`, and `/docs/token_extraction.md` for details.

## Task 14.1: Qwen3-4B Model Setup (Status: DONE)

- The Qwen3-4B model and tokenizer are set up using the Hugging Face `transformers` library.
- Tokenizer loading is used for all data and token extraction workflows.
- Model loading is only performed for inference/evaluation.
- Scripts are compatible with Mac (Apple Silicon) and follow the correct import/module pattern.
- See `.cursor/rules/module_imports.mdc` for enforced import rules.

## Goals

- Build a dataset for spelling, position, and count questions.
- Train and fine-tune a Qwen3-4B model using Unsloth and LoRA optimizations.
- Evaluate if spelling training improves model performance on position and count metrics using a true hold-out set.
- **Clarification:** Qwen3-4B is always used in non-thinking mode (enable_thinking=False). Spelling data is used for training, but evaluation is strictly limited to character position and character count tasks. Spelling is never used as an evaluation metric. All evaluation metrics, scripts, and documentation must focus exclusively on position and count. Any use of thinking mode is prohibited and enforced in code.

## Project Structure

- `configs/` — Configuration files including templates
- `data/` — Training data and generated examples
  - `processed/` — Generated examples and variations
  - `raw/` — Original word lists and data
  - `splits/` — Train/val/test splits
- `docs/` — Project documentation
  - `templates.md` — Template system documentation
  - `data_format.md` — Data format specifications
  - See [Qwen3-4B Tokenizer Compatibility Audit](docs/templates.md#qwen3-4b-tokenizer-compatibility-audit-2024-06-12), [Data Validation](docs/data_format.md#qwen3-4b-tokenizer-compatibility-and-data-validation-2024-06-12), and [Token Extraction Audit](docs/token_extraction.md#qwen3-4b-tokenizer-compatibility-audit-2024-06-12) for audit and validation details.
- `results/` — Analysis results and visualizations
  - `token_analysis/` — Template analysis results
    - `data/` — Raw analysis data in CSV format
    - `figures/` — Generated plots and visualizations
    - `reports/` — HTML analysis reports
- `scripts/` — Utility scripts, PRD, and complexity reports
- `src/` — Source code
  - `analysis/` — Analysis scripts and utilities
    - `template_analysis.py` — Template pattern analysis
    - `template_performance.py` — Performance metrics analysis
    - `visualization_utils.py` — Shared plotting utilities
  - `data/` — Data processing and example generation
  - `evaluation/` — Model evaluation code
  - `models/` — Model definitions
  - `training/` — Training utilities
- `tasks/` — Taskmaster-generated task files and subtasks
- `.env.example` — Template for required environment variables
- `.env` — Your local environment configuration (not committed)
- `README.md` — This documentation

## Getting Started

> **Mac M1/M2 Users:** For a detailed, up-to-date setup guide (including troubleshooting and performance tips), see [docs/apple_silicon_setup.md](docs/apple_silicon_setup.md).
>
> **Local Quantized Inference:** For step-by-step instructions and troubleshooting for Ollama, see [docs/ollama_integration.md](docs/ollama_integration.md).
>
> **Cloud Training & GPU Workflows:** For a complete guide to running this project in the cloud (Colab, Lightning, custom VM), see [docs/cloud_workflow.md](docs/cloud_workflow.md).
>
> **Google Colab Notebooks:** For detailed instructions on running project notebooks in Google Colab, see [docs/colab.md](docs/colab.md).

### 1. Environment Setup (Mac/Apple Silicon)

**Use [uv](https://github.com/astral-sh/uv) for Python environment and package management.**

#### a. Install uv (if not already installed)

```sh
curl -fsSL https://astral.sh/uv/install.sh | bash
```

#### b. Create a virtual environment

```sh
uv venv .venv
```

#### c. Activate the virtual environment

- On macOS/Linux:

  ```sh
  source .venv/bin/activate
  ```

- On Windows:

  ```sh
  .venv\Scripts\activate
  ```

#### d. Install core development dependencies

For a new repo, start with the most common tools:

```sh
uv pip install black ruff mypy ipython requests torch transformers datasets wandb dspy lightning matplotlib seaborn pandas jupyter notebook ipywidgets
```

**Do NOT install Unsloth or xformers locally. These are for cloud environments only.**

#### e. Install Ollama for local quantized inference

```sh
uv pip install ollama
```

- Use Ollama for local quantized inference only. Do not use for training or fine-tuning.

#### f. Freeze installed packages for reproducibility

```sh
uv pip freeze > requirements.txt
```

Commit `requirements.txt` to version control.

#### g. Document the uv version

```sh
uv --version
```

Add the output to your `README.md` or a setup section for future reference.

---

**Tip:**  
Whenever you add new packages, always run `uv pip freeze > requirements.txt` again to keep your requirements up to date.

---

### 2. Cloud Workflow (Google Colab, Lightning, etc.)

**All fine-tuning, Unsloth, and xformers steps must be performed in a cloud environment.**

#### a. Recommended cloud platforms

- Google Colab (Pro/Pro+ recommended for longer jobs)
  - See [Colab Setup Guide](docs/colab.md) for detailed notebook setup
  - Includes repository setup, Python path configuration, and best practices
  - Essential for running project notebooks with GPU acceleration
- Lightning AI
- Any cloud VM with CUDA-enabled GPU

#### b. Cloud-specific setup

```sh
pip install torch transformers datasets wandb dspy lightning matplotlib seaborn pandas jupyter notebook ipywidgets unsloth xformers
```

- Only install `unsloth` and `xformers` in the cloud.
- Use the latest compatible versions of all packages.
- Follow platform-specific instructions for GPU/driver setup.

#### c. Fine-tuning and training

- All model fine-tuning and heavy training must be done in the cloud.
- Use Unsloth and xformers for efficient training.
- Never attempt to install or run Unsloth/xformers locally on Mac/Apple Silicon.

#### d. Data Preparation

- Data preparation and token extraction can be performed locally using Hugging Face transformers, or in the cloud as needed.

---

### 3. Local vs. Cloud Workflow Comparison

| Step                        | Local (Mac/Apple Silicon)         | Cloud (Colab/Lightning/VM)         |
|-----------------------------|-----------------------------------|------------------------------------|
| Python env setup            | uv, .venv                         | pip, conda, or platform default    |
| Install transformers        | ✅                                 | ✅                                 |
| Install Ollama              | ✅                                 | 🚫                                 |
| Install Unsloth/xformers    | 🚫                                 | ✅                                 |
| Data preparation            | ✅                                 | ✅                                 |
| Token extraction            | ✅                                 | ✅                                 |
| Fine-tuning/training        | 🚫                                 | ✅                                 |
| Quantized inference         | ✅ (Ollama)                        | ✅ (if needed)                     |
| GPU acceleration            | 🚫                                 | ✅                                 |

**Legend:** ✅ = Supported, 🚫 = Not supported

---

### 4. Troubleshooting & Warnings

- **Do NOT install Unsloth or xformers locally.** These packages are not compatible with Mac/Apple Silicon and are only for cloud environments with CUDA GPUs.
- If you see errors related to CUDA, xformers, or Unsloth on your Mac, you are likely trying to run a cloud-only step locally. Switch to a cloud environment.
- Use `uv pip install ollama` only on Mac/Apple Silicon for local quantized inference.
- For any issues with package versions, check the [requirements.txt](requirements.txt) and ensure you are using the correct environment.
- If you encounter issues with Hugging Face transformers, ensure you are using version 4.51.0 or higher.
- For cloud GPU setup, refer to your platform's documentation for driver and CUDA installation.

---

### 5. Example Commands

**Local (Mac/Apple Silicon):**

```sh
uv pip install transformers ollama
python scripts/extract_english_tokens.py
```

**Cloud (Colab):**

```sh
pip install transformers unsloth xformers
python scripts/train.py --model Qwen3-4B --data data/processed/tokens.json --output results/model/
```

---

### 6. Environment Variables

- Copy `.env.example` to `.env`:

  ```sh
  cp .env.example .env
  ```

- Fill in all required values (API keys, etc.).
- Never commit `.env` with secrets to version control.

### 7. Authentication

- Log in to Weights & Biases:

  ```sh
  wandb login
  ```

- Log in to Hugging Face:

  ```sh
  huggingface-cli login
  ```

### 8. Project Tasks & Workflow

- All development is managed via Taskmaster tasks.
- To see current tasks:

  ```sh
  task-master list --with-subtasks
  ```

- To see the next actionable task:

  ```sh
  task-master next
  ```

- Tasks are broken down into subtasks for clarity and iterative progress.
- Follow the details and test strategies in each task file in `tasks/`.

### 9. Dataset Creation

The project uses a template-based system to generate diverse training examples. All tokenization and example generation use the Qwen3-4B tokenizer and the English-only token subset.

#### Template System

- Located in `configs/templates/categories.json`
- Multiple template categories (spelling_first, word_first)
- Various styles (simple, playful, educational)
- Configurable token separation (space, comma, dash, etc.)
- **Robust template variable handling:** Templates are parsed for all variables, and the generator fills them dynamically. See [docs/templates.md](docs/templates.md).
- **No answer leakage:** Spelling/structured templates only use {word} in the input; the output is always generated as the letter sequence. See [docs/data_format.md](docs/data_format.md).
- **Separator style mixing:** For each template, the generator produces examples for all separator styles, ensuring variety in the output.

#### Example Generation

```python
from src.data.example_generator import ExampleGenerator, TemplateConfig
from pathlib import Path

# Configure paths
config = TemplateConfig(
    templates_dir=Path("configs/templates"),
    output_dir=Path("data/processed/template_variations")
)

generator = ExampleGenerator(config)

# Generate examples
examples = generator.generate_examples(
    words=["apple", "banana"],
    num_variations=3,
    balance_categories=True
)
# Each example will use a different template and separator style.
```

#### Data Loading and Batching

```python
from src.data.data_loader import TemplateDataLoader, BatchConfig
from pathlib import Path

# Configure batch settings
batch_config = BatchConfig(
    batch_size=32,
    max_length=512,
    similar_length_tolerance=50,
    shuffle=True
)

# Initialize data loader
loader = TemplateDataLoader(
    data_dir=Path("data/processed/template_variations"),
    batch_config=batch_config,
    split_ratios=(0.8, 0.1, 0.1)  # train/val/test
)

# Get dataset statistics
stats = loader.get_stats()
print(f"Total examples: {stats.total_examples}")
print(f"Average sequence length: {stats.avg_sequence_length:.2f}")

# Iterate over batches
for batch in loader.train_batches():
    inputs = batch["inputs"]           # List of input sequences
    outputs = batch["outputs"]         # List of target outputs
    template_cats = batch["template_categories"]  # Template categories
    separator_styles = batch["separator_styles"]  # Separator styles used
```

The data loader provides:

- Efficient memory usage through lazy loading
- Smart batching by grouping similar-length sequences
- Automatic train/val/test splitting
- Performance monitoring and statistics
- Comprehensive batch information including metadata

#### Data Organization

- Training set: Generated from templates with vocabulary words
- Validation/Test sets: Generated from external word lists
- Examples saved in JSON format with metadata
- See `docs/data_format.md` for detailed specifications
- **The canonical English token list is now `data/processed/english_tokens.json` (JSON with a `tokens` key).**

### 10. Analysis Tools

All analysis scripts use the Qwen3-4B tokenizer and support only non-thinking mode. See `/docs/analysis.md` for details.

#### Template Analysis (`src/analysis/template_analysis.py`)

Analyzes characteristics and distribution of template variations:

```python
# Run template analysis
python -m src.analysis.template_analysis \
  --data-dir data/processed \
  --output-dir results/token_analysis \
  --batch-size 32
```

Generates:

- Pattern distribution analysis
- Template length statistics
- Complexity score distributions
- HTML reports with visualizations

#### Performance Analysis (`src/analysis/template_performance.py`)

Analyzes how different template variations affect model performance:

```python
# Run performance analysis
python -m src.analysis.template_performance \
  --data-dir data/processed \
  --output-dir results/token_analysis \
  --batch-size 32
```

Generates:

- Performance metrics by template pattern
- Performance metrics by sequence length
- Confusion matrices
- HTML reports with visualizations

Both scripts use shared visualization utilities from `src/analysis/visualization_utils.py` for consistent styling and report generation.

### 11. Model Training & Evaluation

- All training and evaluation use Qwen3-4B and the English-only token subset.
- **Qwen3-4B is always used in non-thinking mode (enable_thinking=False). Thinking mode is strictly prohibited and enforced in code.**
- Select the correct mode (non-thinking only) and configure sampling parameters as described above.
- For Unsloth-based fine-tuning or any GPU-dependent workflow, use [Google Colab](https://colab.research.google.com/) or [Lightning.ai](https://lightning.ai/lars/home).
- See the section below for cloud workflow instructions.
- Experiments are tracked with W&B.
- **Evaluation focuses only on position and count question metrics, using the hold-out set for true generalization measurement. Spelling is never used as an evaluation metric.**

---

## ⚠️ Mac vs. Cloud GPU Workflow

### Local Mac (Apple Silicon) Environment

- Only install and use packages that are compatible with Mac and do not require a GPU (no Unsloth, no xformers).
- Do all data preparation, code development, and CPU-based evaluation locally.
- If you see errors about `xformers` or `unsloth` during install, **ignore them locally** and move to the cloud workflow for those steps.

### Cloud GPU (Colab/Lightning) Environment for Unsloth

- For any fine-tuning or training that requires Unsloth, LoRA, or GPU acceleration:
  1. Open [Google Colab](https://colab.research.google.com/) or [Lightning.ai](https://lightning.ai/lars/home).
  2. Upload your code and data, or clone your repo.
  3. In a Colab cell, run:

     ```python
     !pip install unsloth torch transformers datasets wandb dspy lightning matplotlib seaborn pandas jupyter notebook ipywidgets
     ```

  4. Proceed with Unsloth-based fine-tuning and training as described in your project tasks.
  5. Download results/models back to your local machine as needed.

---

## Troubleshooting & FAQ

- If dependencies fail to install, ensure you are using uv and not pip directly.
- If you encounter missing environment variables, check `.env.example` for required keys.
- For Taskmaster issues, see the [Taskmaster documentation](https://github.com/roochat/task-master-ai) or run `task-master --help`.
- If you see errors about `xformers` or `unsloth` on Mac, ignore them and use the cloud workflow for those steps.
- If `pip` or `python` commands fail, check that you are using the correct virtual environment:

  ```sh
  which python
  which pip
  ```

  Both should point to your `.venv` directory.

## Directory Reference

- `configs/templates/` — Template configuration files
- `data/processed/template_variations/` — Generated examples
- `docs/templates.md` — Template system documentation
- `docs/data_format.md` — Data format specifications
- `results/token_analysis/` — Analysis results and reports
- `src/analysis/` — Analysis scripts and utilities
- `src/data/example_generator.py` — Example generation code
- `src/data/token_separator.py` — Token separation utilities
- `scripts/` — Scripts, PRD, complexity reports
- `tasks/` — Task files and subtasks
- `.env.example` — Environment variable template
- `.env` — Local environment (not committed)
- `README.md` — Project documentation

## Dataset Split Generation (Qwen3-4B English-Only)

This project uses a task-specific dataset split for all experiments:

- **Training set:** Spelling examples only, generated using the Qwen3-4B English-only token subset. Uses only the `spelling_first`, `word_first`, and `structured` template categories.
- **Validation set:** Character count and character position questions only. No spelling examples. Uses the new `char_count_question` and `char_position_question` template categories.
- **Test set:** Same as validation, but with a disjoint set of tokens. No overlap with validation set.

### How splits are generated

Run the orchestration script:

```sh
PYTHONPATH=. python scripts/generate_dataset_splits.py
```

This will:

- Load the canonical token list from `data/processed/english_tokens.json`
- Generate spelling examples for all tokens (training set)
- Randomly split tokens into validation and test sets (no overlap)
- Generate character count and character position questions for validation and test
- Save the splits as:
  - `data/processed/train_spelling.json`
  - `data/processed/val_char_questions.json`
  - `data/processed/test_char_questions.json`

### Template categories

- Spelling: `spelling_first`, `word_first`, `structured`
- Character count: `char_count_question`
- Character position: `char_position_question`

### Output summary

- Training: 3 spelling examples per token (one per spelling category)
- Validation/Test: 1 character count + 2 character position questions per token
- No overlap between validation and test tokens

See `/docs/data_format.md` for the JSON structure of each split.

## Documentation Links (Updated for 2024-06-12)

- [Data Format and Splits](docs/data_format.md): Full details on all dataset splits, spelling separator conventions, token filtering, and example entries.
- [Template System](docs/templates.md): Template categories, robust variable handling, separator style mixing, and authoring guidelines.
- [Token Extraction](docs/token_extraction.md): Canonical English token set, multi-token set, extraction scripts, and validation.
- [Validation and Testing Framework](docs/validation.md): How to validate all datasets and token sets, CLI usage, troubleshooting, and extending validation rules.
- [Analysis Tools](docs/analysis.md): How to use the new splits for robust evaluation, including separator conventions and token filtering.

## Validation and Testing Framework

To ensure all generated datasets and token sets are valid and compatible with fine-tuning libraries, use the following tools:

### 1. Alpaca Schema Validation

Run the batch validation script to check all datasets in `data/processed/`:

```sh
python scripts/validate_datasets.py
```
- Checks for required fields, empty strings, non-ASCII characters, and (if available) English-only token subset.
- Prints a summary and details of any invalid examples.

### 2. Token Set Validation

Validate the canonical English token set:
```sh
python src/data/validate_alpaca_schema.py data/processed/english_tokens.json
```
Validate the multi-token word set:
```sh
python src/data/validate_alpaca_schema.py data/processed/english_multi_tokens.json
```
- Ensures all tokens/words are unique, valid, and meet project requirements.

### 3. Integrated Validation in Data Generation

Whenever you generate examples using `ExampleGenerator.save_examples`, Alpaca schema validation is automatically run on the output file. Warnings are printed if any invalid examples are found.

See also:
- [Validation and Testing Framework](docs/validation.md)
- [Data Format Documentation](docs/data_format.md)
- [Cloud Workflow Guide](docs/cloud_workflow.md)

## Alpaca Format Validation

All data loading and evaluation pipelines require datasets to conform to the Alpaca format. Validation is enforced automatically in the code, but you can manually validate datasets using:

```sh
PYTHONPATH=. python scripts/validate_datasets.py
```

- See [docs/data_format.md](docs/data_format.md) for the Alpaca data format specification.
- See [docs/validation.md](docs/validation.md) for validation requirements and usage.
- See [scripts/validate_datasets.py](scripts/validate_datasets.py) for the validation script.

**Note:** Always run scripts in the `scripts/` directory with the correct import path (use `PYTHONPATH=.` or `python -m scripts.<script_name>`) to avoid import errors.
