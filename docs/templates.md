# Template System Documentation

## Overview
The template system provides a flexible way to generate diverse training examples for spelling tasks. It supports multiple template categories, styles, and token separation methods.

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

## Categories
Current template categories include:
- `spelling_first`: Templates that present the spelling before the word
- `word_first`: Templates that present the word before its spelling
- `educational`: Templates with an educational or instructional tone
- `playful`: Templates with a more casual, fun tone

## Styles
Each category supports multiple styles:
- `simple`: Basic, straightforward templates
- `playful`: Fun, engaging templates
- `educational`: Teaching-focused templates
- `formal`: Professional or academic templates

## Token Separation
The system supports various token separation styles:
- None: Letters without separation
- Space: Letters separated by spaces
- Comma: Letters separated by commas
- Dash: Letters separated by dashes
- Dots: Letters separated by dots
- Arrow: Letters separated by arrows

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
