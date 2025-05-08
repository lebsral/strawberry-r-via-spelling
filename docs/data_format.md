# Data Format Documentation

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
