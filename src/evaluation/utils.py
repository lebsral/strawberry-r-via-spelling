"""
Utility functions for the evaluation framework.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

def ensure_dir(path: str) -> None:
    """Ensure a directory exists, creating it if necessary."""
    os.makedirs(path, exist_ok=True)

def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_json(data: Dict[str, Any], path: str, indent: int = 2) -> None:
    """Save data as JSON file."""
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def get_output_path(
    base_dir: str,
    subdir: Optional[str] = None,
    filename: Optional[str] = None
) -> str:
    """Construct output file path."""
    path = Path(base_dir)
    if subdir:
        path = path / subdir
    if filename:
        path = path / filename
    return str(path)

def format_metric_value(value: float, precision: int = 4) -> str:
    """Format metric value for display."""
    return f"{value:.{precision}f}"

def format_error_message(
    prediction: str,
    target: str,
    error_type: str
) -> str:
    """Format error message for logging."""
    return (
        f"Error ({error_type}): "
        f"Prediction '{prediction}' != Target '{target}'"
    )

def batch_generator(items, batch_size: int):
    """Generate batches from a list of items."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def setup_output_dirs(base_dir: str, subdirs: list) -> None:
    """Set up output directory structure."""
    for subdir in subdirs:
        ensure_dir(os.path.join(base_dir, subdir))

def merge_configs(
    base_config: Dict[str, Any],
    override_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Merge base configuration with override configuration."""
    if not override_config:
        return base_config

    merged = base_config.copy()
    for key, value in override_config.items():
        if (
            key in merged and
            isinstance(merged[key], dict) and
            isinstance(value, dict)
        ):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value

    return merged

def validate_config(config: Dict[str, Any], required_keys: list) -> None:
    """Validate configuration has required keys."""
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f"Missing required configuration keys: {', '.join(missing_keys)}"
        )

def format_results_summary(results: Dict[str, Any]) -> str:
    """Format evaluation results for display."""
    summary = ["Evaluation Results:"]
    for metric_name, result in results.items():
        if isinstance(result, dict) and 'value' in result:
            value = format_metric_value(result['value'])
            summary.append(f"- {metric_name}: {value}")
    return "\n".join(summary)

def get_error_summary(errors: list, max_examples: int = 5) -> str:
    """Format error examples for display."""
    if not errors:
        return "No errors found."

    summary = ["Error Examples:"]
    for i, error in enumerate(errors[:max_examples], 1):
        summary.append(
            f"{i}. {error.get('error', 'Unknown error')}\n"
            f"   Prediction: '{error.get('prediction', '')}'\n"
            f"   Target: '{error.get('target', '')}'"
        )

    if len(errors) > max_examples:
        summary.append(f"... and {len(errors) - max_examples} more errors")

    return "\n".join(summary)
