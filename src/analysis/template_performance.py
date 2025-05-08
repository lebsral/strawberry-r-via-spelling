"""
Template Performance Analysis Script

Analyzes how different template variations affect model performance.
"""

import os
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from src.data.data_loader import TemplateDataLoader, BatchConfig
from src.analysis.visualization_utils import (
    plot_performance_comparison,
    plot_heatmap,
    generate_html_report
)

# Set up logging
logger = logging.getLogger(__name__)

# Define template categories
TEMPLATE_CATEGORIES = [
    "spelling_first",
    "structured",
    "word_first"
]

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze template performance")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/token_analysis",
        help="Directory to save analysis results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for data loading"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
    )
    return parser.parse_args()

def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def calculate_performance_metrics(
    prediction: str,
    target: str,
    template_category: str,
    template_style: str
) -> Dict[str, float]:
    """Calculate various performance metrics for a prediction."""
    # For demonstration, calculate simple metrics
    accuracy = float(prediction == target)
    precision = float(prediction == target)  # Simplified for demo
    recall = float(prediction == target)     # Simplified for demo
    f1 = float(prediction == target)         # Simplified for demo

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sequence_length': len(target)
    }

def predict_batch(inputs: List[str]) -> List[str]:
    """Generate predictions for a batch of inputs (dummy implementation)."""
    # For demonstration, just return random categories
    return [random.choice(TEMPLATE_CATEGORIES) for _ in inputs]

def analyze_performance(data_loader: TemplateDataLoader) -> Tuple[pd.DataFrame, np.ndarray]:
    """Analyze performance metrics for different template variations."""
    results = []
    confusion_matrix = np.zeros((len(TEMPLATE_CATEGORIES), len(TEMPLATE_CATEGORIES)))

    logger.info("Analyzing performance...")
    for batch in tqdm(data_loader.test_batches()):
        # Get predictions for this batch
        predictions = predict_batch(batch['inputs'])

        # Calculate metrics
        for i, (pred, true_cat) in enumerate(zip(predictions, batch['template_categories'])):
            metrics = calculate_performance_metrics(
                pred,
                batch['outputs'][i],
                batch['template_categories'][i],
                batch['template_styles'][i]
            )
            results.append({
                'template_category': batch['template_categories'][i],
                'template_style': batch['template_styles'][i],
                **metrics
            })

            # Update confusion matrix
            pred_idx = TEMPLATE_CATEGORIES.index(pred)
            true_idx = TEMPLATE_CATEGORIES.index(true_cat)
            confusion_matrix[true_idx][pred_idx] += 1

    # Normalize confusion matrix
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix = np.divide(confusion_matrix, row_sums, where=row_sums!=0)

    return pd.DataFrame(results), confusion_matrix

def analyze_template_performance(
    data_loader: TemplateDataLoader,
    output_dir: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze template performance patterns.

    Args:
        data_loader: Data loader instance
        output_dir: Directory to save results

    Returns:
        Tuple of (performance by pattern, performance by length)
    """
    logger.info("Analyzing performance...")
    results_df, confusion_matrix = analyze_performance(data_loader)

    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(output_dir) / "data" / f"performance_results_{timestamp}.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path)

    # Group by template pattern
    perf_by_pattern = results_df.groupby('template_category').agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std']
    }).round(3)

    # Group by sequence length bins (handling duplicate edges)
    try:
        length_bins = pd.qcut(results_df['sequence_length'], q=5)
    except ValueError:
        # If we get duplicate edges, try fewer bins or use cut instead
        unique_lengths = results_df['sequence_length'].nunique()
        if unique_lengths < 5:
            length_bins = pd.cut(results_df['sequence_length'], bins=unique_lengths)
        else:
            length_bins = pd.qcut(results_df['sequence_length'], q=5, duplicates='drop')

    perf_by_length = results_df.groupby(length_bins).agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std']
    }).round(3)

    # Generate visualizations
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Plot performance comparisons
    plot_performance_comparison(
        perf_by_pattern,
        title="Performance by Template Pattern",
        output_path=figures_dir / f"pattern_performance_{timestamp}.png"
    )

    plot_performance_comparison(
        perf_by_length,
        title="Performance by Sequence Length",
        output_path=figures_dir / f"length_performance_{timestamp}.png"
    )

    # Plot confusion matrix
    plot_heatmap(
        confusion_matrix,
        labels=TEMPLATE_CATEGORIES,
        title="Template Classification Confusion Matrix",
        output_path=figures_dir / f"confusion_matrix_{timestamp}.png"
    )

    # Generate HTML report
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"performance_report_{timestamp}.html"

    generate_html_report(
        title="Template Performance Analysis",
        sections=[
            {
                "title": "Performance by Template Pattern",
                "content": perf_by_pattern.to_html(),
                "figures": [f"../figures/pattern_performance_{timestamp}.png"]
            },
            {
                "title": "Performance by Sequence Length",
                "content": perf_by_length.to_html(),
                "figures": [f"../figures/length_performance_{timestamp}.png"]
            },
            {
                "title": "Confusion Matrix",
                "content": "Template classification confusion matrix showing prediction accuracy across categories.",
                "figures": [f"../figures/confusion_matrix_{timestamp}.png"]
            }
        ],
        output_path=report_path
    )

    return perf_by_pattern, perf_by_length

def main():
    """Main execution function."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Initializing data loader...")
        batch_config = BatchConfig(batch_size=args.batch_size)
        data_loader = TemplateDataLoader(
            data_dir=args.data_dir,
            batch_config=batch_config
        )

        logger.info("Analyzing performance...")
        perf_by_pattern, perf_by_length = analyze_template_performance(
            data_loader,
            args.output_dir
        )

        logger.info("Analysis complete! Check the output directory for results.")

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
