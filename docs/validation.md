# Validation and Testing Framework

This document describes the validation and testing framework for all datasets and token sets in the project, ensuring compatibility with Qwen3-4B and project requirements.

---

## Overview

The project uses a robust validation framework to guarantee:
- All datasets conform to the Alpaca format and project-specific requirements
- Only valid English tokens (or multi-token words) are used in data splits
- All generated data is compatible with Qwen3-4B fine-tuning libraries

Validation is performed using dedicated scripts and validator classes:
- **AlpacaSchemaValidator**: Checks dataset examples for required fields, correct word usage, and canonical token set membership
- **EnglishTokenSetValidator**: Validates the canonical English token set (`english_tokens.json`)
- **EnglishMultiTokenSetValidator**: Validates the multi-token word set (`english_multi_tokens.json`)

---

## Alpaca Schema Validation

To validate all Alpaca-format datasets (e.g., `train_spelling.json`, `val_char_questions.json`, etc.):

```sh
python scripts/validate_datasets.py
```

- Checks for required fields: `instruction`, `input`, `output`
- Ensures the `{word}` in each example is present in the canonical English token set
- Reports missing fields, invalid words, and other format errors
- Prints a summary and detailed error messages for each file

---

## Token Set Validation

To validate the canonical English token set:

```sh
python src/data/validate_alpaca_schema.py data/processed/english_tokens.json
```

- Ensures all tokens are strictly alphabetic, unique, and non-empty

To validate the multi-token word set:

```sh
python src/data/validate_alpaca_schema.py data/processed/english_multi_tokens.json
```

- Ensures all words are multi-token (per Qwen3-4B tokenizer), unique, and non-empty

---

## CLI Usage Examples

- Validate a single dataset file:
  ```sh
  python src/data/validate_alpaca_schema.py data/processed/train_spelling.json --english-tokens data/processed/english_tokens.json
  ```
- Validate all datasets in a directory:
  ```sh
  python src/data/validate_alpaca_schema.py data/processed/ --english-tokens data/processed/english_tokens.json
  ```
- Validate token sets (see above)

---

## Interpreting Results

- **Summary**: Each validator prints a summary of valid/invalid entries and detailed errors
- **Common errors**:
  - Missing required fields
  - `{word}` not in canonical token set
  - Duplicated or non-alphabetic tokens (for token sets)
- **Fixes**: Regenerate data using the latest scripts, or update templates/token sets as needed

---

## Adding New Validation Rules

- Extend the relevant validator class in `src/data/validate_alpaca_schema.py`
- Add new checks for fields, token properties, or format requirements
- Update scripts and documentation to reference new rules

---

## References

- [docs/data_format.md](data_format.md): Data format specifications
- [docs/templates.md](templates.md): Template system and variable handling
- [docs/token_extraction.md](token_extraction.md): Token extraction and canonical sets
- [src/data/validate_alpaca_schema.py](../src/data/validate_alpaca_schema.py): Validator implementation
- [scripts/validate_datasets.py](../scripts/validate_datasets.py): Batch validation script 
