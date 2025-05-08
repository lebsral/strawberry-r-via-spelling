"""
Template Analysis Script

Analyzes characteristics and distribution of template variations.
"""

import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.data_loader import TemplateDataLoader, BatchConfig
from src.analysis.visualization_utils import (
    plot_distribution,
    plot_category_counts,
    plot_heatmap,
    generate_html_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_token_patterns(templates: List[str]) -> Dict[str, int]:
    """
    Analyze token separation patterns in templates.

    Args:
        templates: List of template strings

    Returns:
        Dict mapping pattern types to their counts
    """
    patterns = {}

    for template in templates:
        # Identify separator pattern (space, comma, etc.)
        if "," in template:
            pattern = "comma-separated"
        elif "|" in template:
            pattern = "pipe-separated"
        elif ";" in template:
            pattern = "semicolon-separated"
        else:
            pattern = "space-separated"

        patterns[pattern] = patterns.get(pattern, 0) + 1

    return patterns

def calculate_complexity_score(input_text: str, output_text: str) -> float:
    """Calculate a complexity score for a template based on various factors."""
    score = 0.0

    # Length-based complexity
    input_length = len(input_text)
    output_length = len(output_text)
    length_ratio = output_length / input_length if input_length > 0 else 1
    score += min(length_ratio, 1.0) * 0.3  # 30% weight for length ratio

    # Structural complexity
    structural_score = 0.0
    structural_score += input_text.count('{') * 0.1  # Template variables
    structural_score += input_text.count('|') * 0.1  # Alternatives
    structural_score += input_text.count('?') * 0.1  # Optional elements
    score += min(structural_score, 0.4)  # 40% weight for structural complexity

    # Output complexity
    output_score = 0.0
    output_score += output_text.count('\n') * 0.05  # Multi-line outputs
    output_score += sum(1 for c in output_text if c.isupper()) / len(output_text) * 0.1  # Capitalization
    output_score += len(set(output_text)) / len(output_text) * 0.1  # Character diversity
    score += min(output_score, 0.3)  # 30% weight for output complexity

    return min(score, 1.0)  # Ensure score is between 0 and 1

def analyze_templates(data_loader: TemplateDataLoader) -> Tuple[Dict[str, int], List[int], List[float]]:
    """Analyze patterns and characteristics of templates."""
    patterns = defaultdict(int)
    lengths = []
    complexity_scores = []

    logger.info("Analyzing templates...")
    for batch in tqdm(data_loader.train_batches()):
        # Extract template patterns
        for category in batch['template_categories']:
            patterns[category] += 1

        # Calculate sequence lengths
        lengths.extend([len(x) for x in batch['inputs']])

        # Calculate complexity scores
        for input_text, output_text in zip(batch['inputs'], batch['outputs']):
            score = calculate_complexity_score(input_text, output_text)
            complexity_scores.append(score)

    return dict(patterns), lengths, complexity_scores

def generate_analysis_report(
    patterns: Dict[str, int],
    lengths: List[float],
    complexity_scores: List[float],
    output_dir: str
) -> None:
    """
    Generate visualizations and HTML report.

    Args:
        patterns: Dict of pattern types and counts
        lengths: List of template lengths
        complexity_scores: List of complexity scores
        output_dir: Output directory for results
    """
    logger.info("Generating report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directories
    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    pattern_plot = figures_dir / f"pattern_distribution_{timestamp}.png"
    length_plot = figures_dir / f"length_distribution_{timestamp}.png"
    complexity_plot = figures_dir / f"complexity_distribution_{timestamp}.png"

    plot_category_counts(
        patterns,
        title="Template Pattern Distribution",
        output_path=pattern_plot
    )

    plot_distribution(
        lengths,
        title="Template Length Distribution",
        xlabel="Length",
        ylabel="Count",
        output_path=length_plot,
        bins=30
    )

    plot_distribution(
        complexity_scores,
        title="Template Complexity Distribution",
        xlabel="Complexity Score",
        ylabel="Count",
        output_path=complexity_plot,
        bins=20
    )

    # Calculate summary statistics
    length_stats = {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths)
    }

    complexity_stats = {
        'mean': np.mean(complexity_scores),
        'std': np.std(complexity_scores),
        'min': np.min(complexity_scores),
        'max': np.max(complexity_scores)
    }

    # Generate HTML report
    report_dir = Path(output_dir) / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"template_analysis_{timestamp}.html"

    generate_html_report(
        title="Template Analysis Report",
        sections=[
            {
                "title": "Template Pattern Distribution",
                "content": f"""
                <p>Distribution of template patterns across the dataset:</p>
                <ul>
                {"".join(f"<li>{k}: {v}</li>" for k, v in patterns.items())}
                </ul>
                """,
                "figures": [f"../figures/pattern_distribution_{timestamp}.png"]
            },
            {
                "title": "Template Length Analysis",
                "content": f"""
                <p>Statistics for template lengths:</p>
                <ul>
                <li>Mean: {length_stats['mean']:.2f}</li>
                <li>Standard Deviation: {length_stats['std']:.2f}</li>
                <li>Range: {length_stats['min']:.0f} - {length_stats['max']:.0f}</li>
                </ul>
                """,
                "figures": [f"../figures/length_distribution_{timestamp}.png"]
            },
            {
                "title": "Template Complexity Analysis",
                "content": f"""
                <p>Statistics for template complexity scores:</p>
                <ul>
                <li>Mean: {complexity_stats['mean']:.2f}</li>
                <li>Standard Deviation: {complexity_stats['std']:.2f}</li>
                <li>Range: {complexity_stats['min']:.2f} - {complexity_stats['max']:.2f}</li>
                </ul>
                """,
                "figures": [f"../figures/complexity_distribution_{timestamp}.png"]
            }
        ],
        output_path=report_path
    )

    # Save raw data
    data_dir = Path(output_dir) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        'pattern_type': list(patterns.keys()),
        'count': list(patterns.values())
    }).to_csv(data_dir / f"pattern_counts_{timestamp}.csv", index=False)

    pd.DataFrame({
        'length': lengths,
        'complexity': complexity_scores
    }).to_csv(data_dir / f"template_metrics_{timestamp}.csv", index=False)

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze template variations")
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

def setup_logging(log_level: str) -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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

        logger.info("Analyzing templates...")
        patterns, lengths, complexity_scores = analyze_templates(data_loader)

        logger.info("Generating report...")
        generate_analysis_report(
            patterns,
            lengths,
            complexity_scores,
            args.output_dir
        )

        logger.info("Analysis complete! Check the output directory for results.")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise

if __name__ == "__main__":
    main()
