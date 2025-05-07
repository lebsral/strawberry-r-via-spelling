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

## Validation Results

### Token Statistics

Based on the validation analysis performed in `notebooks/token_validation.ipynb`:

- Total Tokens: 46,789
- Length Distribution:
  - Average: 5.91 characters
  - Median: 6.0 characters
  - Minimum: 2 characters
  - Maximum: 32 characters

### Sample Tokens by Length

Representative samples from the extracted tokens:

- 2 characters: IS, di, Mi, LT, NY
- 3 characters: but, Und, TSA, Joh, aja
- 4 characters: scen, Corp, Anna, lime, mend
- 5 characters: Canad, ultra, wered, COURT, local
- 6 characters: Fallon, romptu, irming, strong, clitor
- 7 characters: ongoing, require, Lindsey, Balkans, holding
- 8 characters: ensional, toddlers, probably, resemble, pointers
- 9 characters: installed, shameless, Sovereign, Permanent, consisted
- 10 characters: aggravated, complicate, threatened, reassuring, ecological

### Validation Process

The validation process includes:

1. Data structure verification
2. Statistical analysis of token lengths
3. Distribution visualization
4. Sample token extraction for manual review

### Results Location

- Token length distribution plot: `results/token_analysis/token_length_distribution.png`
- Detailed analysis results: `results/token_analysis/analysis_results.json`

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
