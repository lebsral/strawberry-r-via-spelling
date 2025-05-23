# Analysis Tools Documentation

> **Environment Workflow Notice:**
>
> - **Analysis scripts can be run locally (Mac/Apple Silicon) or in the cloud.**
> - **Unsloth and xformers are cloud-only**: Never install or use these packages locally. They require CUDA GPUs and are not compatible with Mac/Apple Silicon.
> - **Ollama is for local quantized inference only**: Only install and use Ollama on Mac/Apple Silicon for local inference. Do not use Ollama in the cloud.
> - For full workflow details and troubleshooting, see the [README](../README.md#cloud-workflow-google-colab-lightning-etc) and [Local vs. Cloud Workflow Comparison](../README.md#3-local-vs-cloud-workflow-comparison).
>
> **Example (Local):**
>
> ```sh
> uv pip install transformers ollama
> python -m src.analysis.template_analysis --data-dir data/processed --output-dir results/token_analysis --batch-size 32
> ollama run qwen3-4b:quantized --input data/processed/template_variations/examples.json
> ```
>
> **Example (Cloud):**
>
> ```sh
> pip install transformers unsloth xformers
> python -m src.analysis.template_analysis --data-dir data/processed --output-dir results/token_analysis --batch-size 32
> ```
>
> **Troubleshooting:**
>
> - If you see errors about CUDA, xformers, or Unsloth on Mac, you are trying to run a cloud-only step locally. Switch to a cloud environment.
> - If you see errors about Ollama in the cloud, remove it and use only for local inference.

This document describes the analysis tools available in the project for understanding template patterns and performance metrics.

**Clarification:** Spelling data is used for training, but evaluation is strictly limited to character position and character count tasks. Spelling is never used as an evaluation metric. All evaluation metrics, scripts, and documentation must focus exclusively on position and count.

**Project Policy:** Qwen3-4B is always used in non-thinking mode (enable_thinking=False) for all analysis and evaluation. Thinking mode is strictly prohibited and enforced in code. Any attempt to use thinking mode will raise an error.

**Task 15 (Qwen3-4B compatibility migration) is complete. All code, data, and documentation have been audited and updated to enforce non-thinking mode. All subtasks are done or cancelled as appropriate.**
**CI/CD safeguards for thinking mode are not implemented and not planned. Enforcement is via code and documentation only.**

**Audit (2024-06-11):** The codebase, documentation, and configuration were audited. No references to thinking mode remain except as explicit prohibitions. All analysis tools, scripts, and documentation are compliant. See the README for summary and policy details.

## Updated Workflow (Post-Task 14.1)

- All analysis scripts use the Qwen3-4B **tokenizer** by default for data and template analysis.
- The **full model** is only loaded for inference or evaluation, not for data prep or analysis.
- Scripts and imports must follow the `src/` layout. See `.cursor/rules/module_imports.mdc` for enforced import rules.
- This is the project standard after Task 14.1 (DONE).

## Overview

All analysis scripts and evaluation tools in this project are designed for the Qwen3-4B model, using only the English-only token subset. Analyses must account for both thinking and non-thinking modes, as well as the specific sampling parameters required for Qwen3-4B.

## Tokenizer and Token Subset

- All input and output sequences for analysis must be tokenized with Qwen3-4B.
- Only tokens present in the English-only subset (see `english_tokens.json`) are considered valid for analysis.
- Any analysis of token distribution, sequence length, or template patterns must use this subset.

## Mode-Specific Evaluation

- Qwen3-4B supports two modes:
  - **Thinking Mode:** Model generates intermediate reasoning before the final answer. Use sampling parameters: Temperature=0.6, TopP=0.95, TopK=20, MinP=0.
  - **Non-Thinking Mode:** Model generates direct answers without intermediate reasoning.
- All evaluation scripts must:
  - Specify the mode for each experiment.
  - Parse and separate thinking content from final output when in thinking mode.
  - Compare performance metrics across both modes.

## Template Analysis

### Purpose

The template analysis script (`src/analysis/template_analysis.py`) analyzes the characteristics and distribution of template variations in the dataset.

### Features

- Pattern distribution analysis
- Template length statistics
- Complexity score calculations
- Automated report generation

### Usage

```bash
python -m src.analysis.template_analysis \
  --data-dir data/processed \
  --output-dir results/token_analysis \
  --batch-size 32 \
  --log-level INFO
```

### Arguments

- `--data-dir`: Directory containing processed data (default: "data/processed")
- `--output-dir`: Directory to save analysis results (default: "results/token_analysis")
- `--batch-size`: Batch size for data loading (default: 32)
- `--log-level`: Logging level (default: "INFO")

### Output Structure

```
results/token_analysis/
├── data/
│   ├── pattern_counts_TIMESTAMP.csv
│   └── template_metrics_TIMESTAMP.csv
├── figures/
│   ├── pattern_distribution_TIMESTAMP.png
│   ├── length_distribution_TIMESTAMP.png
│   └── complexity_distribution_TIMESTAMP.png
└── reports/
    └── template_analysis_TIMESTAMP.html
```

## Performance Analysis

### Purpose

The performance analysis script (`src/analysis/template_performance.py`) analyzes how different template variations affect model performance.

### Features

- Performance metrics by template pattern
- Performance metrics by sequence length
- Confusion matrix visualization
- Automated report generation

### Usage

```bash
python -m src.analysis.template_performance \
  --data-dir data/processed \
  --output-dir results/token_analysis \
  --batch-size 32 \
  --log-level INFO
```

### Arguments

- `--data-dir`: Directory containing processed data (default: "data/processed")
- `--output-dir`: Directory to save analysis results (default: "results/token_analysis")
- `--batch-size`: Batch size for data loading (default: 32)
- `--log-level`: Logging level (default: "INFO")

### Output Structure

```
results/token_analysis/
├── data/
│   └── performance_results_TIMESTAMP.csv
├── figures/
│   ├── pattern_performance_TIMESTAMP.png
│   ├── length_performance_TIMESTAMP.png
│   └── confusion_matrix_TIMESTAMP.png
└── reports/
    └── performance_report_TIMESTAMP.html
```

## Visualization Utilities

The `src/analysis/visualization_utils.py` module provides shared plotting functions and consistent styling for visualizations.

### Key Functions

- `plot_distribution()`: Plot histograms of distributions
- `plot_category_counts()`: Plot bar charts of category counts
- `plot_performance_comparison()`: Plot performance metrics comparisons
- `plot_heatmap()`: Plot heatmaps (e.g., confusion matrices)
- `generate_html_report()`: Generate HTML reports with embedded figures

### Styling

- Uses updated seaborn style ('seaborn-v0_8')
- Consistent color palettes via 'husl'
- Standard figure sizes and formatting
- Responsive HTML report design

## Template Categories

The analysis scripts work with three main template categories:

1. `spelling_first` - Templates that present spelling tasks first
2. `structured` - Templates with structured format
3. `word_first` - Templates that present the word first

## Performance Metrics

The performance analysis calculates several metrics:

- Accuracy
- Precision
- Recall
- F1 Score
- Sequence length correlations

## Best Practices

1. **Data Organization**
   - Keep processed data in `data/processed/`
   - Store results in `results/token_analysis/`
   - Use meaningful timestamps in filenames

2. **Analysis Workflow**
   - Run template analysis first to understand data characteristics
   - Follow with performance analysis to evaluate model behavior
   - Review HTML reports for comprehensive insights

3. **Customization**
   - Adjust batch sizes based on available memory
   - Set appropriate logging levels for debugging
   - Modify visualization parameters as needed

4. **Report Interpretation**
   - Check pattern distributions for dataset balance
   - Look for correlations between complexity and performance
   - Identify areas for template improvement

## Future Improvements

Potential areas for enhancement:

- Additional performance metrics
- Interactive visualizations
- Real-time analysis capabilities
- Custom template category support
- Advanced statistical analysis

## Example Analysis Workflow

1. **Run Template Analysis**

   ```sh
   python -m src.analysis.template_analysis \
     --data-dir data/processed \
     --output-dir results/token_analysis \
     --batch-size 32
   ```

   - Ensure all scripts use Qwen3-4B tokenizer and reference `english_tokens.json`.

2. **Run Performance Analysis**

   ```sh
   python -m src.analysis.template_performance \
     --data-dir data/processed \
     --output-dir results/token_analysis \
     --batch-size 32 \
     --mode thinking  # or --mode non-thinking
   ```

   - Compare metrics for both modes.

## Reporting

- All reports and visualizations should indicate the mode (thinking/non-thinking) and confirm use of the English-only token subset.
- Include mode-specific performance breakdowns and error analyses.

## References

- See `docs/token_extraction.md` for token extraction methodology.
- See `docs/data_format.md` for data format specifications.
- See Taskmaster tasks #14 and #15 for migration and compatibility conversion.

## Evaluation Metrics

- Only position and character count metrics are used for evaluation.
- Spelling is never used as an evaluation metric.
- **Qwen3-4B is always used in non-thinking mode.**

## Using the New Dataset Splits for Analysis

With the Qwen3-4B migration, all analysis should use the new dataset splits:

- `data/processed/train_spelling.json`: Spelling examples for training only
- `data/processed/val_char_questions.json`: Character count and character position questions for validation
- `data/processed/test_char_questions.json`: Character count and character position questions for testing

The splits are generated by `scripts/generate_dataset_splits.py` and are fully documented in `/docs/data_format.md`.

- **Training set:** Use only for spelling-based training analysis
- **Validation/Test sets:** Use only for character count and character position evaluation
- There is no overlap between validation and test tokens

Update any analysis scripts to reference these files and to expect the new example types and template categories.

## Analyzing Multi-Token Evaluation Splits

For more challenging evaluation, use the multi-token validation and test splits:

- `data/processed/val_char_questions_multi_token.json`
- `data/processed/test_char_questions_multi_token.json`

These splits are generated from words in `words_alpha.txt` that tokenize to at least 2 tokens (with at least one token ≥2 characters) using the Qwen3-4B tokenizer. The same template logic is used as for the main splits.

- See `/docs/data_format.md` for details on the generation and structure of these splits.
- These splits are for evaluation only and are not used for training.

Update your analysis scripts to include these files for a more robust assessment of model performance on multi-token words.

## Spelling Example Separator Rules (Updated 2024-06-12)

- All spelling examples use exactly one separator between each letter, randomly chosen per example from: space, comma+space, dash, ellipsis, or arrow.
- For commas, the separator is ', ' (comma and single space). For other separators, no extra spaces are used.
- No double or mixed separators are allowed. No run-together letters.
- The separator style is recorded in the metadata for each example as 'separator_style'.
- All tokens are lowercased and filtered to only those present in words_alpha.txt.
- The script enforces these rules and regeneration is always consistent with the latest code.

- See [data_format.md](data_format.md) and [templates.md](templates.md) for full details.

## Data and Token Set Validation (Prerequisite)

Before running any analysis scripts, validate all datasets and token sets using the [validation framework](validation.md):
- Run `python scripts/validate_datasets.py` to check all Alpaca-format datasets
- Run `python src/data/validate_alpaca_schema.py data/processed/english_tokens.json` and `data/processed/english_multi_tokens.json` to check token sets
- See [docs/validation.md](validation.md) for details and troubleshooting
