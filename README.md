# LLM Spelling Project

## Project Overview

This project explores whether training a language model (LLM) on spelling tasks improves its ability to answer position and count questions about words. The experiment is structured using Taskmaster for rigorous, task-driven development and reproducibility.

## Goals
- Build a dataset for spelling, position, and count questions.
- Train and fine-tune a GPT-2 model (or similar) using Unsloth and LoRA optimizations.
- Evaluate if spelling training improves model performance on position and count metrics using a true hold-out set.

## Project Structure
- `scripts/` — Utility scripts, PRD, and complexity reports.
- `tasks/` — Taskmaster-generated task files and subtasks.
- `.env.example` — Template for required environment variables.
- `.env` — Your local environment configuration (not committed).
- `READMD.md` — This documentation.

## Getting Started

### 1. Environment Setup

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
uv pip install black ruff mypy ipython requests
```
Add any project-specific packages as you go (e.g., `torch`, `transformers`, `pandas`, etc.).

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
- Training set: Derived from the tokenizer vocabulary (universal set).
- Validation/Test sets: Derived from external word lists (hold-out sets), ensuring no overlap with training.
- The split is **source-based**, not percentage-based.
- Scripts and notebooks for dataset creation and analysis are in `scripts/`.

### 6. Model Training & Evaluation
- Fine-tuning is performed using Unsloth and LoRA for memory efficiency.
- Experiments are tracked with W&B.
- Evaluation focuses on position and count question metrics, using the hold-out set for true generalization measurement.

###  Contributing
- Follow the Taskmaster workflow for all new features or changes.
- Add new tasks or subtasks as needed using Taskmaster commands.
- Update this README as the project evolves.

## Troubleshooting & FAQ
- If dependencies fail to install, ensure you are using uv and not pip directly.
- If you encounter missing environment variables, check `.env.example` for required keys.
- For Taskmaster issues, see the [Taskmaster documentation](https://github.com/roochat/task-master-ai) or run `task-master --help`.

## Directory Reference
- `scripts/` — Scripts, PRD, complexity reports
- `tasks/` — Task files and subtasks
- `.env.example` — Environment variable template
- `.env` — Local environment (not committed)
- `READMD.md` — Project documentation

## Contact
For questions or contributions, open an issue or contact the project maintainer.
