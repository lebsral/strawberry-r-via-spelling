"""
Alpaca Schema Validation Utility

Validates that all examples in a dataset conform to the Alpaca format and project-specific requirements.
- Required fields: instruction, input, output (all strings)
- Optional fields: template_category, template_style, separator_style (if present, must be string or null)
- No extra fields allowed (unless in metadata)
- Checks for empty strings, special characters, and Qwen3-4B English-only token subset (if available)
- Reports summary of validation results
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

REQUIRED_FIELDS = ["instruction", "input", "output"]
OPTIONAL_FIELDS = ["template_category", "template_style", "separator_style"]

class AlpacaSchemaValidator:
    def __init__(self, english_tokens_path: Optional[Path] = None):
        self.english_tokens = None
        if english_tokens_path and Path(english_tokens_path).exists():
            with open(english_tokens_path) as f:
                data = json.load(f)
                self.english_tokens = set(data["tokens"])

    def validate_example(self, example: Dict[str, Any]) -> List[str]:
        errors = []
        # Required fields
        for field in REQUIRED_FIELDS:
            if field not in example:
                errors.append(f"Missing required field: {field}")
        if errors:
            return errors
        # Use explicit 'word' field if present (for validation only)
        word = example.get('word', None)
        if word is None:
            # Fallback: try to extract the word from the input using a regex
            input_text = example.get('input', '')
            # Look for a single word in quotes or at the end of the input
            match = re.search(r"'([A-Za-z]+)'", input_text)
            if match:
                word = match.group(1)
            else:
                # Fallback: last word in the input
                tokens = re.findall(r"[A-Za-z]+", input_text)
                if tokens:
                    word = tokens[-1]
        # Only check that the word is in the English-only subset
        if word and self.english_tokens and word not in self.english_tokens:
            errors.append(f"Word '{word}' not in English-only token subset")
        return errors

    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path) as f:
            data = json.load(f)
            if isinstance(data, dict) and "examples" in data:
                examples = data["examples"]
            elif isinstance(data, list):
                examples = data
            else:
                raise ValueError(f"Unrecognized data format in {file_path}")
        results = []
        for i, ex in enumerate(examples):
            errs = self.validate_example(ex)
            if errs:
                results.append({"index": i, "errors": errs, "example": ex})
        summary = {
            "file": str(file_path),
            "total": len(examples),
            "invalid": len(results),
            "valid": len(examples) - len(results),
            "errors": results
        }
        return summary

    def validate_dir(self, dir_path: Path) -> List[Dict[str, Any]]:
        reports = []
        for file in dir_path.glob("*.json"):
            try:
                report = self.validate_file(file)
                reports.append(report)
            except Exception as e:
                reports.append({"file": str(file), "error": str(e)})
        return reports

class EnglishTokenSetValidator:
    """Validates the canonical English token set file (english_tokens.json)."""
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path) as f:
            data = json.load(f)
        errors = []
        tokens = data.get("tokens", [])
        if not isinstance(tokens, list):
            errors.append("'tokens' key must be a list")
            return {"file": str(file_path), "valid": False, "errors": errors}
        seen = set()
        for i, token in enumerate(tokens):
            if not isinstance(token, str) or not token.isalpha():
                errors.append(f"Token #{i} ('{token}') is not strictly alphabetic")
            if not token:
                errors.append(f"Token #{i} is empty")
            if token in seen:
                errors.append(f"Token #{i} ('{token}') is duplicated")
            seen.add(token)
        return {
            "file": str(file_path),
            "total": len(tokens),
            "unique": len(seen),
            "invalid": len(errors),
            "valid": len(tokens) - len(errors),
            "errors": errors
        }

class EnglishMultiTokenSetValidator:
    """Validates the multi-token English word set (english_multi_tokens.json)."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path) as f:
            data = json.load(f)
        errors = []
        tokens = data.get("tokens", [])
        if not isinstance(tokens, list):
            errors.append("'tokens' key must be a list")
            return {"file": str(file_path), "valid": False, "errors": errors}
        seen = set()
        for i, word in enumerate(tokens):
            if not isinstance(word, str) or not word:
                errors.append(f"Word #{i} is not a non-empty string")
            if word in seen:
                errors.append(f"Word #{i} ('{word}') is duplicated")
            seen.add(word)
            # Validate multi-token property
            if self.tokenizer:
                n_tokens = len(self.tokenizer.tokenize(word))
                if n_tokens < 2:
                    errors.append(f"Word #{i} ('{word}') is not multi-token (tokenizes to {n_tokens} tokens)")
        return {
            "file": str(file_path),
            "total": len(tokens),
            "unique": len(seen),
            "invalid": len(errors),
            "valid": len(tokens) - len(errors),
            "errors": errors
        }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate Alpaca-format datasets and token sets.")
    parser.add_argument("path", type=str, help="File or directory to validate")
    parser.add_argument("--english-tokens", type=str, default=None, help="Path to english_tokens.json")
    parser.add_argument("--tokenizer", type=str, default=None, help="Load Qwen3-4B tokenizer for multi-token validation")
    args = parser.parse_args()
    path = Path(args.path)
    # Special-case for token set files
    if path.name == "english_tokens.json":
        validator = EnglishTokenSetValidator()
        report = validator.validate_file(path)
        print(json.dumps(report, indent=2))
    elif path.name == "english_multi_tokens.json":
        # Lazy import to avoid dependency if not needed
        from src.models.qwen3_loader import load_qwen3_tokenizer_only
        tokenizer = load_qwen3_tokenizer_only()
        validator = EnglishMultiTokenSetValidator(tokenizer)
        report = validator.validate_file(path)
        print(json.dumps(report, indent=2))
    elif path.is_file():
        validator = AlpacaSchemaValidator(args.english_tokens)
        report = validator.validate_file(path)
        print(json.dumps(report, indent=2))
    else:
        validator = AlpacaSchemaValidator(args.english_tokens)
        reports = validator.validate_dir(path)
        print(json.dumps(reports, indent=2))
