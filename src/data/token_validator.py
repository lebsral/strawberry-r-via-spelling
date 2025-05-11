"""
Token Validation for Qwen3-4B Compatibility

This module provides functionality to validate templates and examples
against the Qwen3-4B tokenizer's English-only token subset.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Set
from transformers import AutoTokenizer

class TokenizerValidator:
    """Validates text against Qwen3-4B tokenizer compatibility requirements."""

    def __init__(self, english_tokens_path: Optional[Path] = None):
        """Initialize the validator with English token subset.

        Args:
            english_tokens_path: Path to english_tokens.json. If None, uses default location.
        """
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

        # For testing without english_tokens.json, use basic ASCII validation
        self.english_tokens = self._load_english_tokens(english_tokens_path)

        # Cache common separators validation results
        self.valid_separators = {
            " ": True,  # Space is always valid
            ",": True,  # Basic punctuation is valid
            "-": True,
            ".": True,
            "->": True,
        }

    def _load_english_tokens(self, tokens_path: Optional[Path] = None) -> Set[str]:
        """Load the English-only token subset.

        Args:
            tokens_path: Optional path to english_tokens.json

        Returns:
            Set of valid English tokens
        """
        if tokens_path is None:
            # Default to project structure
            tokens_path = Path("data/processed/english_tokens.json")

        try:
            with open(tokens_path) as f:
                data = json.load(f)
                return set(data["tokens"])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            # If file doesn't exist or is invalid, use basic ASCII validation
            print("Warning: english_tokens.json not found or invalid. Using basic ASCII validation.")
            # Get all tokens that are ASCII-only
            vocab = self.tokenizer.get_vocab()
            return {
                token for token in vocab.keys()
                if all(ord(c) < 128 for c in token)
                and not token.startswith("<")
                and not token.endswith(">")
            }

    def is_valid_separator(self, separator: str) -> bool:
        """Check if a separator string is valid for the tokenizer.

        Args:
            separator: The separator string to validate

        Returns:
            True if the separator is valid, False otherwise
        """
        # Special case: empty separator is always valid
        if not separator or separator.isspace():
            return True

        # Use cached result if available
        if separator in self.valid_separators:
            return self.valid_separators[separator]

        # For new separators, check if all characters are ASCII
        is_valid = all(ord(c) < 128 for c in separator)
        self.valid_separators[separator] = is_valid
        return is_valid

    def validate_template(self, template: str) -> bool:
        """Check if a template string is compatible with the tokenizer.

        Args:
            template: The template string to validate

        Returns:
            True if the template is valid, False otherwise
        """
        # Provide all possible variables with sample values
        sample_kwargs = {
            "word": "example",
            "letters": "e x a m p l e",
            "letter": "e",
            "n": 1,
            "ordinal": "1st",
            "ordinal_word": "first"
        }
        try:
            sample_text = template.format(**sample_kwargs)
        except Exception:
            return False
        # For validation, we only care about ASCII compatibility
        return all(ord(c) < 128 for c in sample_text)

    def validate_example(self, example: Dict[str, str]) -> bool:
        """Check if a generated example is compatible with the tokenizer.

        Args:
            example: Dictionary containing the example data

        Returns:
            True if the example is valid, False otherwise
        """
        # For now, just check ASCII compatibility
        return (
            all(ord(c) < 128 for c in example["input"])
            and all(ord(c) < 128 for c in example["output"])
        )

    def get_valid_separators(self) -> List[str]:
        """Get the list of valid separator strings.

        Returns:
            List of separator strings that are compatible with the tokenizer
        """
        return [sep for sep, valid in self.valid_separators.items() if valid]

    def suggest_separator_replacement(self, invalid_separator: str) -> str:
        """Suggest a valid replacement for an invalid separator.

        Args:
            invalid_separator: The invalid separator string

        Returns:
            A valid separator that most closely matches the intent
        """
        # Map common invalid separators to valid alternatives
        replacements = {
            "...": ".",  # Replace ellipsis with period
            "=>": "->",  # Replace fat arrow with thin arrow
            "|": "-",    # Replace pipe with dash
            ";": ",",    # Replace semicolon with comma
        }

        return replacements.get(invalid_separator, " ")  # Default to space
