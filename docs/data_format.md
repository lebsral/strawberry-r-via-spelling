# Data Format Documentation

> **Environment Workflow Notice:**
> - **Data preparation, batching, and example generation can be performed locally (Mac/Apple Silicon) or in the cloud.**
> - **Unsloth and xformers are cloud-only**: Never install or use these packages locally. They require CUDA GPUs and are not compatible with Mac/Apple Silicon.
> - **Ollama is for local quantized inference only**: Only install and use Ollama on Mac/Apple Silicon for local inference. Do not use Ollama in the cloud.
> - For full workflow details and troubleshooting, see the [README](../README.md#cloud-workflow-google-colab-lightning-etc) and [Local vs. Cloud Workflow Comparison](../README.md#3-local-vs-cloud-workflow-comparison).
>
> **Example (Local):**
> ```sh
> uv pip install transformers ollama
> python scripts/data_prep.py --input data/raw/words.txt --output data/processed/template_variations/examples.json
> ollama run qwen3-4b:quantized --input data/processed/template_variations/examples.json
> ```
>
> **Example (Cloud):**
> ```sh
> pip install transformers unsloth xformers
> python scripts/data_prep.py --input data/raw/words.txt --output data/processed/template_variations/examples.json
> ```
>
> **Troubleshooting:**
> - If you see errors about CUDA, xformers, or Unsloth on Mac, you are trying to run a cloud-only step locally. Switch to a cloud environment.
> - If you see errors about Ollama in the cloud, remove it and use only for local inference.

## Overview
This document describes the data formats used in the project for training examples, templates, and configuration files.

## Directory Structure
```
data/
├── processed/
│   └── template_variations/
│       ├── template_examples.json
│       └── diverse_examples.json
├── raw/
└── splits/

configs/
└── templates/
    └── categories.json
```

## Template Configuration Format
Located in `configs/templates/categories.json`:

```json
{
  "spelling_first": {
    "simple": [
      "{letters} — that spells '{word}.'",
      "The letters {letters} spell the word '{word}.'"
    ],
    "playful": [
      "Say it with me: {letters} — {word}!",
      "These little letters — {letters} — team up to make '{word}.'"
    ]
  },
  "word_first": {
    "simple": [
      "The word '{word}' is spelled {letters}.",
      "'{word}' is spelled like this: {letters}"
    ]
  }
}
```

### Template Variables
- `{word}`: Replaced with the target word
- `{letters}`: Replaced with separated letters using the chosen separator style

## Generated Examples Format
Located in `data/processed/template_variations/*.json`:

```json
{
  "examples": [
    {
      "word": "apple",
      "template": "The letters {letters} spell the word '{word}.'",
      "category": "spelling_first",
      "style": "simple",
      "separator_style": "space",
      "generated_text": "The letters a p p l e spell the word 'apple.'"
    }
  ],
  "metadata": {
    "generation_date": "2024-03-20",
    "num_examples": 100,
    "template_version": "1.0",
    "categories_used": ["spelling_first", "word_first"],
    "styles_used": ["simple", "playful", "educational"]
  }
}
```

### Example Fields
- `word`: The target word being spelled
- `template`: The original template string used
- `category`: The template category (e.g., "spelling_first")
- `style`: The template style (e.g., "simple", "playful")
- `separator_style`: The token separation style used
- `generated_text`: The final formatted example

### Metadata Fields
- `generation_date`: When the examples were generated
- `num_examples`: Total number of examples in the file
- `template_version`: Version of the template system used
- `categories_used`: List of template categories used
- `styles_used`: List of template styles used

## Token Separator Styles
Available separator styles for formatting letters:

| Style  | Example         | Description                    |
|--------|----------------|--------------------------------|
| none   | straw         | No separation between letters   |
| space  | s t r a w     | Space between letters          |
| comma  | s, t, r, a, w | Comma and space separation     |
| dash   | s-t-r-a-w     | Dash between letters           |
| dots   | s...t...r...w | Triple dots between letters    |
| arrow  | s->t->r->a->w | Arrow between letters          |

## Configuration Classes

### TemplateConfig
```python
@dataclass
class TemplateConfig:
    templates_dir: Path  # Directory containing template files
    output_dir: Path    # Directory for saving generated examples
```

### SeparatorConfig
```python
@dataclass
class SeparatorConfig:
    style: SeparatorStyle  # Separator style to use
    prefix: str = ""      # Optional prefix before each token
    suffix: str = ""      # Optional suffix after each token
```

## Data Loading and Batching

### BatchConfig
```python
@dataclass
class BatchConfig:
    batch_size: int = 32          # Number of examples per batch
    max_length: int = 512         # Maximum sequence length
    similar_length_tolerance: int = 50  # Max length difference for grouping
    shuffle: bool = True          # Whether to shuffle examples
```

### DataStats
```python
@dataclass
class DataStats:
    total_examples: int           # Total number of examples loaded
    avg_sequence_length: float    # Average sequence length
    length_distribution: Dict[int, int]  # Distribution of sequence lengths
    template_distribution: Dict[str, int]  # Distribution of template types
```

### Batch Format
The data loader yields batches in the following format:
```python
{
    "inputs": List[str],           # Input sequences
    "outputs": List[str],          # Target outputs
    "template_categories": List[str],  # Template categories used
    "template_styles": List[str],   # Template styles
    "separator_styles": List[str],  # Separator styles used
    "metadata": List[Dict]         # Additional example metadata
}
```

### Efficient Loading Features
1. **Lazy Loading**
   - Data is only loaded when first needed
   - Memory-efficient for large datasets

2. **Smart Batching**
   - Groups similar-length sequences together
   - Reduces padding waste
   - Improves training efficiency

3. **Train/Val/Test Splits**
   - Automatic split ratio handling
   - Default: 80% train, 10% val, 10% test
   - Configurable split ratios

4. **Performance Monitoring**
   - Tracks dataset statistics
   - Reports template distribution
   - Monitors sequence lengths

## Best Practices
1. **File Organization**
   - Keep raw data in `data/raw/`
   - Store processed examples in `data/processed/template_variations/`
   - Use descriptive filenames for example sets

2. **Data Validation**
   - Verify template variables are properly replaced
   - Check separator styles are consistently applied
   - Ensure JSON output is properly formatted

3. **Metadata Management**
   - Include generation date and version info
   - Track which templates and styles were used
   - Document any special processing or filtering

4. **Example Generation**
   - Balance categories and styles when possible
   - Vary separator styles for diversity
   - Test with different word lengths
   - Validate output formatting

# Data Format Specifications for Qwen3-4B

## Overview

All data formats in this project are designed for compatibility with the Qwen3-4B tokenizer and use only the English-only token subset (~50k tokens). This ensures that all training, validation, and evaluation are performed exclusively on English tokens, matching the experimental design and model requirements.

## Data Structure

- **Input Sequences:**
  - All input sequences are tokenized using the Qwen3-4B tokenizer.
  - Only tokens present in the English-only subset (see `english_tokens.json`) are allowed.
  - Any data containing non-English tokens is filtered out during preprocessing.

- **Metadata:**
  - Each example includes metadata specifying the template category, separator style, and tokenization details.
  - Metadata fields:
    - `template_category`: e.g., `spelling_first`, `word_first`
    - `separator_style`: e.g., `space`, `comma`, `dash`
    - `tokenizer`: always `Qwen3-4B`
    - `token_subset`: always `english_only`

- **Output Sequences:**
  - Output sequences are also tokenized with Qwen3-4B and restricted to the English-only subset.

## Example JSON Structure

```json
{
  "input": "How do you spell apple?",
  "output": "A P P L E",
  "template_category": "spelling_first",
  "separator_style": "space",
  "tokenizer": "Qwen3-4B",
  "token_subset": "english_only"
}
```

## Data Loading and Batching

- All data loaders and batching utilities must use the Qwen3-4B tokenizer and reference `english_tokens.json` to ensure only valid tokens are processed.
- Batching should group sequences of similar length for efficiency, as before.
- Any sequence containing out-of-vocabulary tokens (not in the English-only subset) should be excluded or flagged.

## Integration with Token Extraction

- The canonical English token list is now `data/processed/english_tokens.json` (JSON with a `tokens` key). The legacy .txt output has been removed.
- All data processing scripts must load and use this subset for filtering and validation.

## Evaluation Considerations

- All evaluation scripts must ensure that both predictions and references use only the English token subset.
- Mode-specific evaluation (thinking/non-thinking) should be documented in `docs/analysis.md`.

## References
- See `docs/token_extraction.md` for extraction methodology.
- See `docs/analysis.md` for evaluation details.

# Data Format Specifications

**Clarification:** Spelling data is used for training, but evaluation is strictly limited to character position and character count tasks. Spelling is never used as an evaluation metric. All evaluation metrics, scripts, and documentation must focus exclusively on position and count.

**Project Policy:** Qwen3-4B is always used in non-thinking mode (enable_thinking=False) for all data processing and evaluation. Thinking mode is strictly prohibited and enforced in code. Any attempt to use thinking mode will raise an error.

**Audit (2024-06-11):** The codebase, documentation, and configuration were audited. No references to thinking mode remain except as explicit prohibitions. All data processing tools, scripts, and documentation are compliant. See the README for summary and policy details.

## Evaluation Fields

- Only position and character count are used for evaluation.
- Spelling is never used as an evaluation metric.
- **Qwen3-4B is always used in non-thinking mode.**

**Task 15 (Qwen3-4B compatibility migration) is complete. All code, data, and documentation have been audited and updated to enforce non-thinking mode. All subtasks are done or cancelled as appropriate.**
**CI/CD safeguards for thinking mode are not implemented and not planned. Enforcement is via code and documentation only.**

## Dataset Split Structure (Qwen3-4B English-Only)

The dataset is split into three files, each with a specific purpose and example type:

### 1. Training Set: Spelling Examples
- **File:** `data/processed/train_spelling.json`
- **Purpose:** Used for model training. Contains only spelling examples.
- **Template categories:** `spelling_first`, `word_first`, `structured`
- **Example entry:**
  ```json
  {
    "input": "The word 'coins' is spelled c, o, i, n, s.",
    "output": "coins",
    "template_category": "word_first",
    "template_style": "simple",
    "separator_style": "comma"
  }
  ```

### 2. Validation Set: Character Count & Position Questions
- **File:** `data/processed/val_char_questions.json`
- **Purpose:** Used for evaluation. Contains only character count and character position questions. No spelling examples.
- **Template categories:** `char_count_question`, `char_position_question`
- **Example entries:**
  ```json
  {
    "input": "How many characters are in 'banana'?",
    "output": "6",
    "template_category": "char_count_question",
    "template_style": "simple",
    "separator_style": null
  }
  {
    "input": "What is the 3th character in 'banana'?",
    "output": "n",
    "template_category": "char_position_question",
    "template_style": "simple",
    "separator_style": null
  }
  ```

### 3. Test Set: Character Count & Position Questions
- **File:** `data/processed/test_char_questions.json`
- **Purpose:** Used for final evaluation. Same structure as validation set, but with a disjoint set of tokens.
- **Template categories:** `char_count_question`, `char_position_question`
- **Example entries:** (see above)

### Regenerating the Splits

To regenerate all splits, run:

```sh
PYTHONPATH=. python scripts/generate_dataset_splits.py
```

This will overwrite the three files above with new splits based on the current token list and templates.

### Notes
- There is no overlap between validation and test tokens.
- Each split is a JSON file with a top-level `examples` key containing a list of example objects.
- See `configs/templates/categories.json` for the full set of templates used.

## Multi-Token Validation/Test Splits

To create more challenging evaluation sets, we generate additional validation and test splits using multi-token words from the [dwyl/english-words](https://github.com/dwyl/english-words) `words_alpha.txt` list.

### Source and Filtering
- **Source:** `data/raw/words_alpha.txt`
- **Filtering criteria:**
  - The word must tokenize to at least 2 tokens using the Qwen3-4B tokenizer.
  - At least one token must be at least 2 characters long (excluding any special tokenization markers).

### Generation Process
- Randomly sample 10,000 words for each split (validation and test).
- For each word, generate character count and character position questions using the same logic as the main splits.
- Validate that all words in the output meet the criteria.

### Output Files
- `data/processed/val_char_questions_multi_token.json`
- `data/processed/test_char_questions_multi_token.json`

### Example Entry
```json
{
  "input": "Which character is at position 2 in 'abandonment'?",
  "output": "b",
  "template_category": "char_position_question",
  "template_style": null,
  "separator_style": null
}
```

### Notes
- These splits are for evaluation only and are not used for training.
- The process is fully reproducible using the script `scripts/generate_dataset_splits.py`.
- The source file `words_alpha.txt` should be kept in `data/raw/` for reproducibility.

## Spelling Example Separator Rules (Updated 2024-06-12)

- Spelling examples in `train_spelling.json` always use exactly one separator between each letter, chosen randomly per example from:
  - space (`c o i n s`)
  - comma+space (`c, o, i, n, s`)
  - dash (`c-o-i-n-s`)
  - ellipsis (`c...o...i...n...s`)
  - arrow (`c->o->i->n->s`)
- For commas, the separator is `, ` (comma and single space). For other separators, no extra spaces are used.
- No double or mixed separators are allowed. No run-together letters.
- The separator style is recorded in the metadata for each example as `separator_style`.
- All tokens are lowercased and filtered to only those present in `words_alpha.txt`.
- The script enforces these rules and regeneration is always consistent with the latest code.

### Example entry (train_spelling.json):
```json
{
  "input": "The word 'coins' is spelled c, o, i, n, s.",
  "output": "coins",
  "template_category": "word_first",
  "template_style": "simple",
  "separator_style": "comma"
}
```

## Qwen3-4B Tokenizer Compatibility and Data Validation (2024-06-12)

### Compatibility Requirements
- All data, templates, and generated examples must use only tokens from the canonical English token list (`data/processed/english_tokens.json`).
- No non-English or non-ASCII tokens are permitted in any generated data or template.
- Separators and template variables are validated for compatibility before use.

### Audit and Validation Process
- The codebase was audited and updated to enforce these requirements using the new `TokenizerValidator` (`src/data/token_validator.py`).
- All templates and separators are checked for compatibility before use in data generation.
- The `ExampleGenerator` (`src/data/example_generator.py`) and `TokenSeparator` (`src/data/token_separator.py`) have been updated to use the validator.
- A comprehensive test script (`scripts/test_tokenizer_compatibility.py`) validates all templates, separators, and generated examples.

### How to Validate
- Run the test script after any changes to templates or tokenization logic:
  ```sh
  python scripts/test_tokenizer_compatibility.py
  ```
- The script will output warnings for any incompatible templates or separators and confirm that only valid examples are produced.

### References
- See also: [docs/templates.md](templates.md), [docs/token_extraction.md](token_extraction.md)

## Alpaca-Style Data Format for Spelling/Structured Examples (2024-06-12)

- **Input**: Always a natural language prompt containing only the {word} variable (never {letters} or the answer).
- **Output**: The letter sequence for the word, using a separator style (space, comma, dash, period, arrow, etc.).
- **No answer leakage**: The input never contains the answer.
- **Separator style mixing**: For each template, the generator produces examples for all separator styles, ensuring variety in the output.

### Example JSON Entry

```json
{
  "instruction": "Spell the word: apple",
  "input": "Spell the word: apple",
  "output": "a p p l e"
}
```

```json
{
  "instruction": "How do you spell 'banana'?",
  "input": "How do you spell 'banana'?",
  "output": "b-a-n-a-n-a"
}
```

```json
{
  "instruction": "Write out the letters in: craft",
  "input": "Write out the letters in: craft",
  "output": "c -> r -> a -> f -> t"
}
```

- See [docs/templates.md](templates.md) for template authoring and variable handling details.
- See [src/data/example_generator.py](../src/data/example_generator.py) for code.

---

## Validation/Evaluation Splits
- Validation and test splits use only character count and character position templates (never spelling).
- See the README and this file's earlier sections for split details.

---
