# LLM Spelling Project

## Project Overview

This project explores whether training a language model (LLM) on spelling tasks improves its ability to answer position and count questions about words. The experiment is structured using Taskmaster for rigorous, task-driven development and reproducibility.

## Goals
- Build a dataset for spelling, position, and count questions.
- Train and fine-tune a GPT-2 model (or similar) using Unsloth and LoRA optimizations.
- Evaluate if spelling training improves model performance on position and count metrics using a true hold-out set.

## Project Structure
- `configs/` — Configuration files including templates
- `data/` — Training data and generated examples
  - `processed/` — Generated examples and variations
  - `raw/` — Original word lists and data
  - `splits/` — Train/val/test splits
- `docs/` — Project documentation
  - `templates.md` — Template system documentation
  - `data_format.md` — Data format specifications
- `scripts/` — Utility scripts, PRD, and complexity reports
- `src/` — Source code
  - `data/` — Data processing and example generation
  - `evaluation/` — Model evaluation code
  - `models/` — Model definitions
  - `training/` — Training utilities
- `tasks/` — Taskmaster-generated task files and subtasks
- `.env.example` — Template for required environment variables
- `.env` — Your local environment configuration (not committed)
- `README.md` — This documentation

## Getting Started

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
**Do NOT install Unsloth or xformers locally.**

#### e. Freeze installed packages for reproducibility
```sh
uv pip freeze > requirements.txt
```
Commit `requirements.txt` to version control.

#### f. Document the uv version
```sh
uv --version
```
Add the output to your `README.md` or a setup section for future reference.

---

**Tip:**  
Whenever you add new packages, always run `uv pip freeze > requirements.txt` again to keep your requirements up to date.

### 2. Environment Variables
- Copy `.env.example` to `.env`:
  ```sh
  cp .env.example .env
  ```
- Fill in all required values (API keys, etc.).
- Never commit `.env` with secrets to version control.

### 3. Authentication
- Log in to Weights & Biases:
  ```sh
  wandb login
  ```
- Log in to Hugging Face:
  ```sh
  huggingface-cli login
  ```

### 4. Project Tasks & Workflow
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

### 5. Dataset Creation
The project uses a template-based system to generate diverse training examples:

#### Template System
- Located in `configs/templates/categories.json`
- Multiple template categories (spelling_first, word_first)
- Various styles (simple, playful, educational)
- Configurable token separation (space, comma, dash, etc.)

#### Example Generation
```python
from src.data.example_generator import ExampleGenerator, TemplateConfig
from pathlib import Path

# Configure paths
config = TemplateConfig(
    templates_dir=Path("configs/templates"),
    output_dir=Path("data/processed/template_variations")
)

# Initialize generator
generator = ExampleGenerator(config)

# Generate examples
examples = generator.generate_examples(
    words=["apple", "banana"],
    num_variations=3,
    balance_categories=True
)
```

#### Data Organization
- Training set: Generated from templates with vocabulary words
- Validation/Test sets: Generated from external word lists
- Examples saved in JSON format with metadata
- See `docs/data_format.md` for detailed specifications

### 6. Model Training & Evaluation
- **Local Mac:** Only run code that does not require GPU, Unsloth, or xformers.
- **For Unsloth-based fine-tuning or any GPU-dependent workflow:**
  - Use [Google Colab](https://colab.research.google.com/) or [Lightning.ai](https://lightning.ai/lars/home).
  - See the section below for cloud workflow instructions.
- Experiments are tracked with W&B.
- Evaluation focuses on position and count question metrics, using the hold-out set for true generalization measurement.

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
- `src/data/example_generator.py` — Example generation code
- `src/data/token_separator.py` — Token separation utilities
- `scripts/` — Scripts, PRD, complexity reports
- `tasks/` — Task files and subtasks
- `.env.example` — Environment variable template
- `.env` — Local environment (not committed)
- `README.md` — Project documentation

## Contribution Guidelines
- Follow the Taskmaster workflow for all new features or changes.
- Add new tasks or subtasks as needed using Taskmaster commands.
- Update this README as the project evolves.
- Use clear commit messages and document any changes to onboarding or environment setup.
- Reference code style and best practices as needed.

## Contact
For questions or contributions, open an issue or contact the project maintainer.
