# Analysis Tools Documentation

This document describes the analysis tools available in the project for understanding template patterns and performance metrics.

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
