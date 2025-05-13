"""
Validate all datasets in the data/processed directory.
Handles both Alpaca-format training data and token list files.

Usage:
    python -m scripts.validate_datasets

This script will:
1. Validate english_tokens.json and english_multi_tokens.json using their specific validators
2. Validate all other .json files as Alpaca-format training data
3. Display a summary of validation results
"""

import os
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any
from contextlib import contextmanager

from src.data.validate_alpaca_schema import (
    AlpacaSchemaValidator,
    EnglishTokenSetValidator,
    EnglishMultiTokenSetValidator
)
from src.models.qwen3_loader import load_qwen3_tokenizer_only

@contextmanager
def progress_tracker(message: str, end_message: str = "Done!"):
    """Context manager to track progress of long-running operations."""
    print(message)
    sys.stdout.flush()
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        print(f"{end_message} (took {duration:.1f}s)")
        sys.stdout.flush()
    except Exception as e:
        print(f"Error after {time.time() - start_time:.1f}s: {str(e)}")
        sys.stdout.flush()
        raise

def print_validation_report(report: Dict[str, Any]) -> None:
    """Print a validation report in a clear, formatted way."""
    filename = Path(report["file"]).name
    print(f"\nFile: {filename}")
    sys.stdout.flush()

    if "error" in report:
        print(f"  Error: {report['error']}")
        sys.stdout.flush()
        return

    if filename in ["english_tokens.json", "english_multi_tokens.json"]:
        print(f"  Total tokens: {report.get('total', 0)}")
        print(f"  Unique tokens: {report.get('unique', 0)}")
        print(f"  Valid: {report.get('valid', 0)}")
        print(f"  Invalid: {report.get('invalid', 0)}")
        if report.get("errors", []):
            print("\n  Errors:")
            for error in report["errors"]:
                print(f"    - {error}")
        sys.stdout.flush()
    else:
        print(f"  Total examples: {report.get('total', 0)}")
        print(f"  Valid: {report.get('valid', 0)}")
        print(f"  Invalid: {report.get('invalid', 0)}")
        if report.get("errors", []):
            print("\n  Errors:")
            for error in report["errors"]:
                print(f"    Example {error['index']}:")
                for err in error["errors"]:
                    print(f"    - {err}")
        sys.stdout.flush()

def main():
    # Setup paths
    print("\nInitializing validation process...")
    sys.stdout.flush()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    english_tokens_path = data_dir / "english_tokens.json"

    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Found {len(list(data_dir.glob('*.json')))} JSON files to validate")
    sys.stdout.flush()

    print("\n=== Validating Datasets ===\n")
    sys.stdout.flush()

    # First validate english_tokens.json
    with progress_tracker("Step 1: Validating english_tokens.json...", "Completed english_tokens.json validation"):
        if english_tokens_path.exists():
            print("  - Found english_tokens.json")
            sys.stdout.flush()
            validator = EnglishTokenSetValidator()
            report = validator.validate_file(english_tokens_path)
            print_validation_report(report)
        else:
            print("  - english_tokens.json not found, skipping")
            sys.stdout.flush()

    # Then validate english_multi_tokens.json
    print("\nStep 2: Validating english_multi_tokens.json...")
    sys.stdout.flush()
    multi_tokens_path = data_dir / "english_multi_tokens.json"
    if multi_tokens_path.exists():
        print("  - Found english_multi_tokens.json")
        sys.stdout.flush()

        with progress_tracker("  - Loading Qwen3 tokenizer (this may take several minutes)...", "Tokenizer loaded"):
            tokenizer = load_qwen3_tokenizer_only()

        with progress_tracker("  - Validating file...", "File validation complete"):
            validator = EnglishMultiTokenSetValidator(tokenizer)
            report = validator.validate_file(multi_tokens_path)
            print_validation_report(report)
    else:
        print("  - english_multi_tokens.json not found, skipping")
        sys.stdout.flush()

    # Finally validate all other JSON files as Alpaca format
    print("\nStep 3: Validating Alpaca format files...")
    sys.stdout.flush()

    with progress_tracker("Loading Alpaca validator...", "Alpaca validator ready"):
        alpaca_validator = AlpacaSchemaValidator(english_tokens_path)

    other_files = [f for f in data_dir.glob("*.json")
                   if f.name not in ["english_tokens.json", "english_multi_tokens.json"]]
    print(f"  - Found {len(other_files)} Alpaca format files to validate")
    sys.stdout.flush()

    for i, file in enumerate(other_files, 1):
        with progress_tracker(f"\n  Processing file {i}/{len(other_files)}: {file.name}", f"Completed {file.name}"):
            report = alpaca_validator.validate_file(file)
            print_validation_report(report)

    print("\n=== Validation Complete ===")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
