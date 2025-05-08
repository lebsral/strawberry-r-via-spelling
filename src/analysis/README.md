# Template Analysis Scripts

This directory contains Python scripts for analyzing template variations and their impact on model performance.

## Scripts

### `template_analysis.py`

Analyzes template characteristics and patterns:
- Token separation patterns
- Template length distribution
- Template complexity metrics

Usage:
```bash
python template_analysis.py [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--batch-size BATCH_SIZE]
```

Arguments:
- `--data-dir`: Directory containing processed data (default: "data/processed")
- `--output-dir`: Directory to save analysis results (default: "results/token_analysis")
- `--batch-size`: Batch size for data loading (default: 32)

### `template_performance.py`

Analyzes how different template variations affect model performance:
- Performance by template pattern
- Performance by template length
- Metric correlations

Usage:
```bash
python template_performance.py --model-path MODEL_PATH [--data-dir DATA_DIR] [--output-dir OUTPUT_DIR] [--batch-size BATCH_SIZE]
```

Arguments:
- `--model-path`: Path to trained model (required)
- `--data-dir`: Directory containing processed data (default: "data/processed")
- `--output-dir`: Directory to save analysis results (default: "results/token_analysis")
- `--batch-size`: Batch size for data loading (default: 32)

### `visualization_utils.py`

Shared utilities for visualization and report generation:
- Plot styling and configuration
- Standard plot types (distributions, comparisons, heatmaps)
- HTML report generation

## Output Structure

Results are saved in the specified output directory with the following structure:

```
results/token_analysis/
├── figures/          # PNG/PDF visualizations
│   ├── pattern_distribution_*.png
│   ├── length_distribution_*.png
│   ├── complexity_distribution_*.png
│   ├── pattern_performance_*.png
│   ├── length_performance_*.png
│   └── metrics_heatmap_*.png
├── reports/          # HTML analysis reports
│   ├── template_analysis_*.html
│   └── performance_analysis_*.html
└── data/            # Raw analysis data
    ├── pattern_counts_*.csv
    ├── template_metrics_*.csv
    ├── pattern_performance_*.csv
    └── length_performance_*.csv
```

## Dependencies

Required packages:
- matplotlib>=3.7.1
- seaborn>=0.12.2
- pandas>=2.0.0
- numpy>=1.24.3
- tqdm>=4.65.0
- scikit-learn>=1.2.2

Install dependencies:
```bash
pip install -r requirements.txt
``` 
