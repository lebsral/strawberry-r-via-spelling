"""
Visualization utilities for template analysis.

This module provides shared plotting functions and consistent styling
for template analysis visualizations.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set style defaults
plt.style.use('seaborn-v0_8')  # Updated style name for newer versions
sns.set_palette("husl")

def setup_plot(title: str, xlabel: str, ylabel: str, figsize: tuple = (10, 6)) -> None:
    """Set up a new matplotlib figure with consistent styling."""
    plt.figure(figsize=figsize)
    plt.title(title, pad=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def save_plot(filepath: Union[str, Path], dpi: int = 300, bbox_inches: str = 'tight') -> None:
    """Save the current plot with standard parameters."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    plt.close()

def plot_distribution(
    data: List[Union[int, float]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Union[str, Path],
    bins: int = 30
) -> None:
    """Plot a histogram of the distribution."""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()

def plot_category_counts(
    categories: Dict[str, int],
    title: str,
    output_path: Union[str, Path]
) -> None:
    """Plot a bar chart of category counts."""
    plt.figure(figsize=(12, 6))
    categories_sorted = dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))

    plt.bar(categories_sorted.keys(), categories_sorted.values())
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_performance_comparison(
    data: pd.DataFrame,
    title: str,
    output_path: Union[str, Path]
) -> None:
    """Plot performance metrics comparison."""
    plt.figure(figsize=(12, 6))

    # Extract mean values and standard deviations
    means = data.xs('mean', axis=1, level=1)
    stds = data.xs('std', axis=1, level=1)

    x = np.arange(len(means.index))
    width = 0.35

    # Plot bars for accuracy and f1
    plt.bar(x - width/2, means['accuracy'], width, label='Accuracy',
            yerr=stds['accuracy'], capsize=5)
    plt.bar(x + width/2, means['f1'], width, label='F1 Score',
            yerr=stds['f1'], capsize=5)

    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Score")
    plt.xticks(x, means.index, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_heatmap(
    matrix: np.ndarray,
    labels: List[str],
    title: str,
    output_path: Union[str, Path]
) -> None:
    """Plot a heatmap (e.g., for confusion matrices)."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_html_report(
    title: str,
    sections: List[Dict[str, Any]],
    output_path: Union[str, Path]
) -> None:
    """
    Generate an HTML report with embedded figures.

    Args:
        title: Report title
        sections: List of dictionaries with keys:
            - title: Section title
            - content: HTML content or text
            - figures: List of paths to figures to include
        output_path: Where to save the HTML report
    """
    html_content = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        f"<title>{title}</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 40px; }",
        "h1 { color: #2c3e50; }",
        "h2 { color: #34495e; margin-top: 30px; }",
        "img { max-width: 100%; height: auto; margin: 20px 0; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #f5f5f5; }",
        "tr:nth-child(even) { background-color: #f9f9f9; }",
        "</style>",
        "</head>",
        "<body>",
        f"<h1>{title}</h1>"
    ]

    for section in sections:
        html_content.extend([
            f"<h2>{section['title']}</h2>",
            f"<div>{section['content']}</div>"
        ])

        if 'figures' in section:
            for fig_path in section['figures']:
                html_content.append(f'<img src="{fig_path}" alt="Figure">')

    html_content.extend([
        "</body>",
        "</html>"
    ])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_content))
