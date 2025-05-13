# Qwen3-4B Token Extraction and English-Only Subset

> **Environment Workflow Notice:**
>
> - **Token extraction and data preparation can be performed locally (Mac/Apple Silicon) or in the cloud.**
> - **Unsloth and xformers are cloud-only**: Never install or use these packages locally. They require CUDA GPUs and are not compatible with Mac/Apple Silicon.
> - **Ollama is for local quantized inference only**: Only install and use Ollama on Mac/Apple Silicon for local inference. Do not use Ollama in the cloud.
> - For full workflow details and troubleshooting, see the [README](../README.md#cloud-workflow-google-colab-lightning-etc) and [Local vs. Cloud Workflow Comparison](../README.md#3-local-vs-cloud-workflow-comparison).
>
> **Example (Local):**
>
> ```sh
> uv pip install transformers ollama
> python scripts/token_extraction.py --model Qwen3-4B --input data/raw/words.txt --output data/processed/tokens.json
> ollama run qwen3-4b:quantized --input data/processed/tokens.json
> ```
>
> **Example (Cloud):**
>
> ```sh
> pip install transformers unsloth xformers
> python scripts/token_extraction.py --model Qwen3-4B --input data/raw/words.txt --output data/processed/tokens.json
> ```
>
> **Troubleshooting:**
>
> - If you see errors about CUDA, xformers, or Unsloth on Mac, you are trying to run a cloud-only step locally. Switch to a cloud environment.
> - If you see errors about Ollama in the cloud, remove it and use only for local inference.

**Project Policy:** Qwen3-4B is always used in non-thinking mode (enable_thinking=False) for all token extraction and evaluation. Thinking mode is strictly prohibited and enforced in code. Any attempt to use thinking mode will raise an error.

**Task 15 (Qwen3-4B compatibility migration) is complete. All code, data, and documentation have been audited and updated to enforce non-thinking mode. All subtasks are done or cancelled as appropriate.**
**CI/CD safeguards for thinking mode are not implemented and not planned. Enforcement is via code and documentation only.**

**Audit (2024-06-11):** The codebase, documentation, and configuration were audited. No references to thinking mode remain except as explicit prohibitions. All token extraction tools, scripts, and documentation are compliant. See the README for summary and policy details.

**Clarification:** Spelling data is used for training, but evaluation is strictly limited to character position and character count tasks. Spelling is never used as an evaluation metric. All evaluation metrics, scripts, and documentation must focus exclusively on position and count.

## Current Workflow (Post-Task 14.1)

- The Qwen3-4B **tokenizer** is loaded by default for all data preparation, token extraction, and analysis tasks.
- The **full model** is only loaded for inference or evaluation, not for data prep.
- Scripts and imports must follow the `src/` layout. See `.cursor/rules/module_imports.mdc` for enforced import rules.
- This setup is now the project standard (Task 14.1: DONE).

## Component Token Extraction (New)

The project now includes sophisticated handling of multi-word tokens and their component parts:

### Multi-Token Word Processing

1. **Initial Token Extraction**
   - Extract all English tokens using `extract_english_tokens.py`
   - Identify multi-word tokens (e.g., "overboard", "underestimate")
   - Store results in `data/processed/english_tokens.json`

2. **Component Token Analysis**
   - Use `extract_component_tokens.py` to:
     - Break down multi-word tokens into components
     - Track usage frequency of each component
     - Identify common prefixes/suffixes
   - Results stored in `data/processed/component_tokens.json`

3. **Dataset Generation**
   - Generate examples based on component token frequency
   - More examples for frequently used components
   - Balanced distribution across template categories

### Component Token Statistics

- Total unique component tokens: 2,716
- Most common components:
  - "over" (1090 uses)
  - "less" (921 uses)
  - "out" (633 uses)

### Example Usage

```python
# Extract component tokens and analyze usage
python scripts/extract_component_tokens.py

# Generate dataset splits using component token information
python scripts/generate_dataset_splits.py
```

## Overview

Qwen3-4B's tokenizer supports multiple languages, but only about 50,000 tokens are English. For this project, we extract and use only the English tokens for all training and evaluation.

## Why English-Only?

- The experiment focuses on English spelling, position, and count tasks.
- Using only English tokens ensures the model is not biased by non-English vocabulary and improves evaluation clarity.

## Why Use Hugging Face Transformers?

- **Tokenizer-Only Workflows:** All data preparation and token extraction use the Qwen3-4B tokenizer from `transformers` to ensure compatibility and reproducibility.
- **Model Compatibility:** Ensures all data, templates, and evaluation are aligned with the Qwen3-4B vocabulary and tokenization logic.
- **Cross-Platform:** Works on Mac/Apple Silicon, Linux, and cloud environments.
- **Community Standard:** Well-documented, widely used, and actively maintained.

## Version Compatibility

- **transformers >= 4.51.0** is required for Qwen3-4B support and Apple Silicon compatibility.
- Always check your installed version:

  ```sh
  python -c 'import transformers; print(transformers.__version__)'
  ```

## Step-by-Step: Extracting English-Only Tokens

### 1. Load the Qwen3-4B Tokenizer

```python
from transformers import AutoTokenizer

# Use the official Hugging Face repo or your local path
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
```

### 2. Identify English Tokens

```python
import re
english_token_pattern = re.compile(r'^[A-Za-z]+$')
english_tokens = [token for token in qwen_tokenizer.get_vocab() if english_token_pattern.match(token)]
```

### 3. Save the English Token Subset

```python
import json
with open('english_tokens.json', 'w') as f:
    json.dump({'tokens': english_tokens}, f)
```

## Example: Data Preparation with transformers

### Tokenizing a List of Words

```python
words = ["apple", "banana", "strawberry"]
encodings = qwen_tokenizer(words, padding=True, truncation=True, return_tensors="pt")
print(encodings.input_ids)
```

### Expected Input/Output Formats

- **Input:** List of words or sentences (e.g., from `data/raw/words.txt`)
- **Output:** Tokenized IDs, attention masks, or JSON files with token lists (e.g., `data/processed/tokens.json`)

## Using transformers in Scripts

- All scripts in this project that require tokenization or data prep should import and use the Qwen3-4B tokenizer as shown above.
- For batch processing, use the tokenizer's built-in batching and padding features.
- Always validate that all tokens are in the English-only subset for your experiments.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ImportError: No module named 'transformers' | Install with `uv pip install transformers` |
| Version mismatch or missing Qwen3-4B | Upgrade transformers: `uv pip install -U transformers` |
| OOV (out-of-vocabulary) tokens | Ensure you are using the English-only subset and the correct tokenizer version |
| Slow tokenization on large files | Use batch tokenization and avoid unnecessary loops |
| Apple Silicon: torch/transformers install fails | See [apple_silicon_setup.md](apple_silicon_setup.md) |

## References

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/en/index)
- [README: Environment Setup](../README.md#1-environment-setup-macapple-silicon)
- [Apple Silicon Setup Guide](apple_silicon_setup.md)

**This is the canonical reference for all transformers-based data preparation in this project.**

## Scripts and Utilities

- See `scripts/extract_english_tokens.py` for the canonical implementation. All downstream scripts should reference the generated `english_tokens.json` (JSON with a `tokens` key).
- All downstream scripts should reference the generated `english_tokens.json` (JSON with a `tokens` key).

## Related Tasks

- See Taskmaster tasks #14 and #15 for migration and compatibility conversion.

## Evaluation Metrics

- Only position and character count are used for evaluation.
- Spelling is never used as an evaluation metric.
- **Qwen3-4B is always used in non-thinking mode.**

## Qwen3-4B Tokenizer Compatibility Audit (2024-06-12)

### English Token Extraction and Validation

- The canonical English token list is stored in `data/processed/english_tokens.json` (JSON with a `tokens` key).
- All downstream code, templates, and data generation use this file to ensure compatibility with the Qwen3-4B tokenizer.
- The `TokenizerValidator` (`src/data/token_validator.py`) loads this file and validates all templates, separators, and generated examples.

### Validation Process

- The validator checks that all tokens, separators, and template variables are compatible with the English-only token subset.
- If `english_tokens.json` is missing, the validator falls back to ASCII-only validation (with a warning).
- The test script (`scripts/test_tokenizer_compatibility.py`) provides comprehensive validation and should be run after any changes to token extraction or template logic.

### References

- See also: [docs/templates.md](templates.md), [docs/data_format.md](data_format.md)

## Token Set Validation

- The canonical English token set is stored in `data/processed/english_tokens.json` (JSON with a `tokens` key)
- The multi-token word set is stored in `data/processed/english_multi_tokens.json` (JSON with a `tokens` key)
- Both sets are validated using dedicated validators (see [docs/validation.md](validation.md))

### How to Run Validation

- Validate the canonical token set:
  ```sh
  python src/data/validate_alpaca_schema.py data/processed/english_tokens.json
  ```
- Validate the multi-token set:
  ```sh
  python src/data/validate_alpaca_schema.py data/processed/english_multi_tokens.json
  ```

### Troubleshooting
- If validation fails, regenerate the token set using the latest extraction script
- See [docs/validation.md](validation.md) for troubleshooting and advanced usage

### Extending Validation
- To add new rules, extend the relevant validator class in `src/data/validate_alpaca_schema.py`
