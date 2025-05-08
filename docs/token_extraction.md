# GPT-2 Token Extraction Process

This document describes the process of extracting multi-character, letter-based tokens from the GPT-2 vocabulary and the validation of the extracted tokens.

## Overview

The project extracts letter-based tokens from GPT-2's vocabulary to analyze how the model tokenizes words and subwords. This information is crucial for understanding the model's handling of spelling-related tasks.

## Token Extraction Process

### Implementation Details

The token extraction is implemented in `src/data/token_extractor.py` and follows these steps:

1. Load the GPT-2 tokenizer vocabulary
2. Filter tokens based on criteria:
   - Multi-character tokens only (length ≥ 2)
   - Letter-based tokens (containing alphabetic characters)
   - Exclude special tokens and non-word characters

### Output Format

Extracted tokens are saved in JSON format at `data/processed/gpt2_letter_tokens.json` with the following structure:

```json
{
  "tokens": [
    {
      "token": "string",
      "token_id": number,
      "char_length": number
    }
  ]
}
```

## Validation

Based on the validation analysis performed using our template analysis scripts:

1. Most common token patterns are well-represented
2. Token sequence lengths follow expected distributions
3. Template variations maintain consistent structure

### Running Validation

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Run the template analysis script:

```bash
python -m src.analysis.template_analysis \
  --data-dir data/processed \
  --output-dir results/token_analysis \
  --log-level INFO
```

3. Review the generated HTML report in `results/token_analysis/reports/`

### Dependencies

Required packages (installed via requirements.txt):

- seaborn (for visualizations)
- scikit-learn (for metrics)
- matplotlib (for plotting)

## Usage

To reproduce the token extraction and validation:

1. Run the extraction script:

   ```bash
   python src/data/token_extractor.py
   ```

2. Run the validation notebook:

   ```bash
   jupyter notebook notebooks/token_validation.ipynb
   ```

## Dependencies

The token extraction and validation process requires:

- transformers (for GPT-2 tokenizer)
- pandas (for data analysis)
- matplotlib (for visualization)
- seaborn (for enhanced plotting)
- jupyter (for running validation notebook)

## Future Improvements

Potential areas for enhancement:

1. Additional token metadata (e.g., frequency in common text)
2. More detailed statistical analysis
3. Token clustering by patterns
4. Integration with spelling task generation
