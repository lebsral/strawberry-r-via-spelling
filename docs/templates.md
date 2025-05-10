# Template System Documentation (Qwen3-4B)

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

## Validation

- All template-generated examples should be tokenized with Qwen3-4B and checked to ensure all tokens are in the English-only subset.
- If a template or separator produces out-of-vocabulary tokens, update the template or filter the example.

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
