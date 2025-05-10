# Qwen3-4B Token Extraction and English-Only Subset

**Project Policy:** Qwen3-4B is always used in non-thinking mode (enable_thinking=False) for all token extraction and evaluation. Thinking mode is strictly prohibited and enforced in code. Any attempt to use thinking mode will raise an error.

**Clarification:** Spelling data is used for training, but evaluation is strictly limited to character position and character count tasks. Spelling is never used as an evaluation metric. All evaluation metrics, scripts, and documentation must focus exclusively on position and count.

## Current Workflow (Post-Task 14.1)

- The Qwen3-4B **tokenizer** is loaded by default for all data preparation, token extraction, and analysis tasks.
- The **full model** is only loaded for inference or evaluation, not for data prep.
- Scripts and imports must follow the `src/` layout. See `.cursor/rules/module_imports.mdc` for enforced import rules.
- This setup is now the project standard (Task 14.1: DONE).

## Overview

Qwen3-4B's tokenizer supports multiple languages, but only about 50,000 tokens are English. For this project, we extract and use only the English tokens for all training and evaluation.

## Why English-Only?

- The experiment focuses on English spelling, position, and count tasks.
- Using only English tokens ensures the model is not biased by non-English vocabulary and improves evaluation clarity.

## Extraction Methodology

1. **Load the Qwen3-4B Tokenizer**
   - Use the `transformers` library (>=4.51.0) to load the Qwen3-4B tokenizer.
   - Example:
     ```python
     from transformers import AutoTokenizer
     tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
     ```

2. **Identify English Tokens**
   - Iterate through the tokenizer's vocabulary.
   - For each token, check if it consists only of English alphabetic characters (A-Z, a-z) and common English punctuation.
   - Exclude tokens containing non-English characters or symbols.
   - Example:
     ```python
     import re
     english_token_pattern = re.compile(r'^[A-Za-z]+$')
     english_tokens = [token for token in tokenizer.get_vocab() if english_token_pattern.match(token)]
     ```

3. **Save the English Token Subset**
   - Store the filtered English tokens in a JSON file for use in data processing and evaluation.
   - Example:
     ```python
     import json
     with open('english_tokens.json', 'w') as f:
         json.dump(english_tokens, f)
     ```

4. **Integrate with Data Processing**
   - Update all data processing scripts to use only the English token subset for training and evaluation.
   - See `docs/data_format.md` for how this affects data structure.

## Scripts and Utilities

- See `scripts/extract_english_tokens.py` (to be created/updated) for a full implementation.
- All downstream scripts should reference the generated `english_tokens.json`.

## References
- [Qwen3-4B Model Card](https://huggingface.co/Qwen/Qwen3-4B)
- [Unsloth Qwen3-4B Blog](https://unsloth.ai/blog/qwen3)

## Related Tasks
- See Taskmaster tasks #14 and #15 for migration and compatibility conversion.

## Evaluation Metrics

- Only position and character count are used for evaluation.
- Spelling is never used as an evaluation metric.
- **Qwen3-4B is always used in non-thinking mode.**
