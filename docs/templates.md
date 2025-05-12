# Template System Documentation (Qwen3-4B)

> **Environment Workflow Notice:**
>
> - **Template generation and data preparation can be performed locally (Mac/Apple Silicon) or in the cloud.**
> - **Unsloth and xformers are cloud-only**: Never install or use these packages locally. They require CUDA GPUs and are not compatible with Mac/Apple Silicon.
> - **Ollama is for local quantized inference only**: Only install and use Ollama on Mac/Apple Silicon for local inference. Do not use Ollama in the cloud.
> - For full workflow details and troubleshooting, see the [README](../README.md#cloud-workflow-google-colab-lightning-etc) and [Local vs. Cloud Workflow Comparison](../README.md#3-local-vs-cloud-workflow-comparison).
>
> **Example (Local):**
>
> ```sh
> uv pip install transformers ollama
> python scripts/template_generation.py --config configs/templates/categories.json --output data/processed/template_variations/examples.json
> ollama run qwen3-4b:quantized --input data/processed/template_variations/examples.json
> ```
>
> **Example (Cloud):**
>
> ```sh
> pip install transformers unsloth xformers
> python scripts/template_generation.py --config configs/templates/categories.json --output data/processed/template_variations/examples.json
> ```
>
> **Troubleshooting:**
>
> - If you see errors about CUDA, xformers, or Unsloth on Mac, you are trying to run a cloud-only step locally. Switch to a cloud environment.
> - If you see errors about Ollama in the cloud, remove it and use only for local inference.

**Project Policy:** Qwen3-4B is always used in non-thinking mode (enable_thinking=False) for all template processing and evaluation. Thinking mode is strictly prohibited and enforced in code. Any attempt to use thinking mode will raise an error.

**Task 15 (Qwen3-4B compatibility migration) is complete. All code, data, and documentation have been audited and updated to enforce non-thinking mode. All subtasks are done or cancelled as appropriate.**
**CI/CD safeguards for thinking mode are not implemented and not planned. Enforcement is via code and documentation only.**

**Audit (2024-06-11):** The codebase, documentation, and configuration were audited. No references to thinking mode remain except as explicit prohibitions. All template tools, scripts, and documentation are compliant. See the README for summary and policy details.

**Clarification:** Spelling data is used for training, but evaluation is strictly limited to character position and character count tasks. Spelling is never used as an evaluation metric. All evaluation metrics, scripts, and documentation must focus exclusively on position and count.

> **Note:** As of Task 14.1 (DONE), all template-based data generation uses the Qwen3-4B **tokenizer** only. The full model is only loaded for inference/evaluation. Scripts and imports must follow the `src/` layout (see `.cursor/rules/module_imports.mdc`). This is now the project standard.

## Overview

All template generation and token separation strategies in this project are designed for compatibility with the Qwen3-4B tokenizer and the English-only token subset. This ensures that all generated examples are valid for the model and experiment.

## Template Structure

Templates are organized in `configs/templates/categories.json` with the following hierarchy:

```json
{
  "category_name": {
    "style_name": [
      "template string 1",
      "template string 2",
      ...
    ]
  }
}
```

### Template Variables

Templates use the following variables:

- `{word}`: The target word being spelled
- `{letters}`: The separated letters of the word

Example:

```
"{letters} — that spells '{word}.'"
```

## Template Categories

- All templates must produce input and output sequences that tokenize cleanly with Qwen3-4B and use only tokens from the English-only subset (see `english_tokens.json`).
- Template categories (e.g., `spelling_first`, `word_first`) remain the same, but all examples must be validated for token compatibility.

## Styles

Each category supports multiple styles:

- `simple`: Basic, straightforward templates
- `playful`: Fun, engaging templates
- `educational`: Teaching-focused templates
- `formal`: Professional or academic templates

## Separator Styles

- Only use separator styles (space, comma, dash, etc.) that are represented in the English-only token subset.
- Avoid using separators or punctuation that are not present in `english_tokens.json`.

## Example Template (Qwen3-4B Compatible)

```json
{
  "input": "How do you spell banana?",
  "output": "B A N A N A",
  "template_category": "spelling_first",
  "separator_style": "space"
}
```

## Validation and Testing

- All templates and generated examples are validated using the [validation framework](validation.md)
- The `{word}` field is included in generated examples for validation (not used in training/inference)
- Only words present in the canonical token set (`english_tokens.json`) or multi-token set (`english_multi_tokens.json`) are used
- Template and separator compatibility is checked before use
- See [docs/validation.md](validation.md) for CLI usage and troubleshooting

### Troubleshooting Template Validation
- If a template or separator fails validation, update or replace it
- Run the test script after any changes:
  ```sh
  python scripts/test_tokenizer_compatibility.py
  ```
- See [docs/data_format.md](data_format.md) and [docs/validation.md](validation.md) for details

## Integration with Data Processing

- Template generation scripts must load and use `english_tokens.json` to validate all generated examples.
- See `docs/data_format.md` for data structure requirements.

## References

- See `docs/token_extraction.md` for token extraction methodology.
- See `docs/data_format.md` for data format specifications.

## Example Generation

The `ExampleGenerator` class handles example generation with features:

- Random template selection within categories
- Style balancing across examples
- Configurable token separation
- Batch example generation
- Category balancing option

### Usage Example

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

## Output Format

Generated examples are saved in JSON format with metadata:

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
  ]
}
```

## Adding New Templates

1. Add new templates to the appropriate category and style in `categories.json`
2. Ensure templates use the standard variables (`{word}`, `{letters}`)
3. Test with the example generator to verify formatting

## Best Practices

- Use natural language patterns
- Include punctuation for readability
- Consider educational value
- Maintain consistent style within categories
- Test with various word lengths

## File Structure

```
configs/
  templates/
    categories.json       # Main template definitions
src/
  data/
    template_processor.py # Template processing logic
    example_generator.py  # Example generation using templates
docs/
  templates.md           # This documentation
  data_format.md         # Data format specifications
```

## Example Usage

```python
from src.data.template_processor import TemplateProcessor

# Initialize processor
processor = TemplateProcessor('configs/templates/categories.json')

# Generate example with specific template
example = processor.generate_example(
    word="straw",
    category="spelling_first",
    style="playful",
    separator=" "
)

# Generate random variation
random_example = processor.generate_random_example("straw")
```

## Quality Metrics

When creating or modifying templates, ensure they meet these criteria:

1. Clarity: Instructions should be clear and unambiguous
2. Consistency: Follow established patterns within categories
3. Variability: Provide meaningful variations in presentation
4. Formatting: Maintain proper spacing and punctuation
5. Scalability: Work well with words of different lengths

## Evaluation Considerations

- Only position and character count are used for evaluation.
- Spelling is never used as an evaluation metric.
- **Qwen3-4B is always used in non-thinking mode.**

## New Template Categories for Dataset Splits

With the Qwen3-4B migration, two new template categories have been added to support character count and character position questions:

- **char_count_question**: Used for validation and test sets. Example templates:
  - "How many characters are in '{word}'?"
  - "Count the number of characters in '{word}'."
  - "What is the length of '{word}'?"
- **char_position_question**: Used for validation and test sets. Example templates:
  - "What is the {n}th character in '{word}'?"
  - "Which character is at position {n} in '{word}'?"
  - "In '{word}', what letter is number {n}?"

These categories are defined in `configs/templates/categories.json` and are used exclusively for generating validation and test examples. The training set continues to use only spelling-related categories (`spelling_first`, `word_first`, `structured`).

### How these templates are used

- The orchestration script `scripts/generate_dataset_splits.py` generates all splits:
  - **Training:** Spelling examples only (one per spelling category per token)
  - **Validation/Test:** Character count and character position questions only (no spelling)
- The script ensures no overlap between validation and test tokens.

See `/docs/data_format.md` for the structure of each split file and example entries.

## Multi-Token Evaluation Splits

The template system supports generating character count and character position questions for multi-token words, as used in the multi-token validation and test splits. The same template categories and logic are applied as for the main splits.

- See `/docs/data_format.md` for details on the filtering, generation, and structure of these splits.
- These splits are generated using the same orchestration script and template pool, ensuring consistency in evaluation.

## Spelling Template Separator Rules (Updated 2024-06-12)

- Spelling templates always use exactly one separator between each letter, randomly chosen per example from:
  - space (`c o i n s`)
  - comma+space (`c, o, i, n, s`)
  - dash (`c-o-i-n-s`)
  - ellipsis (`c...o...i...n...s`)
  - arrow (`c->o->i->n->s`)
- For commas, the separator is `,` (comma and single space). For other separators, no extra spaces are used.
- No double or mixed separators are allowed. No run-together letters.
- The separator style is recorded in the metadata for each example as `separator_style`.
- All tokens are lowercased and filtered to only those present in `words_alpha.txt`.
- The script enforces these rules and regeneration is always consistent with the latest code.

### Example template output

- Template: `The word '{word}' is spelled {letters}.`
- Output: `The word 'coins' is spelled c, o, i, n, s.`

## Qwen3-4B Tokenizer Compatibility Audit (2024-06-12)

### Overview

All template and example generation code has been audited and updated for compatibility with the Qwen3-4B tokenizer, using the English-only token subset. This ensures that all generated data, templates, and separators are valid for the model and do not include non-English or non-ASCII tokens.

### Key Changes

- **TokenizerValidator** (`src/data/token_validator.py`):
  - Loads `english_tokens.json` (canonical English token list) or falls back to ASCII-only validation.
  - Provides methods to check separator, template, and example compatibility.
- **TokenSeparator** (`src/data/token_separator.py`):
  - Uses the validator to ensure only compatible separators are used.
  - Suggests replacements for invalid separators.
- **ExampleGenerator** (`src/data/example_generator.py`):
  - Validates all templates and generated examples for compatibility.
  - Filters out or warns on templates with missing or incompatible variables.
- **Test Script** (`scripts/test_tokenizer_compatibility.py`):
  - Validates all separator styles, templates, and example generation.
  - Output confirms only compatible examples are produced.

### Validation Process

- All templates and separators are checked for compatibility before use.
- Any template or separator that is not compatible is either filtered out or replaced.
- The test script provides comprehensive validation and should be run after any changes to templates or tokenization logic.

### References

- See also: [docs/data_format.md](data_format.md), [docs/token_extraction.md](token_extraction.md)

## Robust Template Variable Handling (2024-06-12)

- Templates are parsed for all variable placeholders (e.g., {word}, {letters}, {n}, {ordinal_word}, {letter}).
- The generator dynamically fills templates, generating all valid combinations for spelling/structured categories (all input templates × all separator styles).
- **No answer leakage:** Templates for spelling/structured categories must only use {word} in the input. The output is always generated by the code as the letter sequence, using the selected separator style.
- If a template requires variables that cannot be filled (e.g., missing {n} or {letter}), it is skipped with a warning.
- See [src/data/example_generator.py](../src/data/example_generator.py) for implementation details.

### Example Usage

```python
from src.data.example_generator import ExampleGenerator, TemplateConfig
from pathlib import Path

config = TemplateConfig(
    templates_dir=Path("configs/templates"),
    output_dir=Path("data/processed/template_variations")
)
generator = ExampleGenerator(config)
examples = generator.generate_examples(["apple", "banana"], num_variations=3, balance_categories=True)
```

### Template Authoring Guidelines

- Only use {word} in the input for spelling/structured templates.
- Do not include {letters} or the answer in the input.
- For char count/position, use {n}, {ordinal_word}, or {letter} as needed.
- See the code for how variables are detected and filled.

---

## Separator Style Mixing

- For each spelling/structured template, the generator produces examples for all separator styles: space, comma, dash, period, arrow.
- This ensures the model sees a variety of input/output formats.

---

## See Also

- [docs/data_format.md](data_format.md) for data format details
- [src/data/example_generator.py](../src/data/example_generator.py) for code
