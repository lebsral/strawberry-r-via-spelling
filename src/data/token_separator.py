"""
Token Separation Strategy Implementation

This module provides various token separation strategies for formatting spelling examples.
It supports different styles of separating characters/tokens for enhanced learning.
"""

from typing import List, Dict, Optional, Union
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .token_validator import TokenizerValidator

class SeparatorStyle(Enum):
    """Enumeration of available separator styles."""
    NONE = "none"  # No separator
    SPACE = "space"  # Simple space
    COMMA = "comma"  # Comma with space
    DASH = "dash"  # Hyphen
    PERIOD = "period"  # Single period
    ARROW = "arrow"  # Arrow symbol
    CUSTOM = "custom"  # Custom separator

@dataclass
class SeparatorConfig:
    """Configuration for token separation."""
    style: SeparatorStyle
    separator: str
    capitalize: bool = False
    add_spaces: bool = True

    # Define style configurations as a class variable
    STYLE_CONFIGS = {
        SeparatorStyle.NONE: {"separator": "", "add_spaces": False},
        SeparatorStyle.SPACE: {"separator": " ", "add_spaces": False},
        SeparatorStyle.COMMA: {"separator": ",", "add_spaces": True},
        SeparatorStyle.DASH: {"separator": "-", "add_spaces": False},
        SeparatorStyle.PERIOD: {"separator": ".", "add_spaces": False},
        SeparatorStyle.ARROW: {"separator": "->", "add_spaces": True},
        SeparatorStyle.CUSTOM: {"separator": " ", "add_spaces": True},
    }

    @classmethod
    def from_style(cls, style: SeparatorStyle) -> 'SeparatorConfig':
        """Create a separator configuration from a style."""
        config = cls.STYLE_CONFIGS.get(style, cls.STYLE_CONFIGS[SeparatorStyle.SPACE])
        return cls(
            style=style,
            separator=config["separator"],
            add_spaces=config["add_spaces"]
        )

class TokenSeparator:
    """Handles the separation of tokens/characters with various styles."""

    def __init__(self, style_or_config=None, validator=None):
        if style_or_config is None:
            style_or_config = SeparatorStyle.SPACE
        if isinstance(style_or_config, SeparatorConfig):
            self.config = style_or_config
        else:
            self.config = SeparatorConfig.from_style(style_or_config)
        # Use provided validator or create a new one
        if validator is not None:
            self.validator = validator
        else:
            from .token_validator import TokenizerValidator
            self.validator = TokenizerValidator()

        # Validate current separator
        if not self.validator.is_valid_separator(self.config.separator):
            # Get replacement and update config
            replacement = self.validator.suggest_separator_replacement(self.config.separator)
            self.config.separator = replacement
            # Update style if needed
            for style, cfg in SeparatorConfig.STYLE_CONFIGS.items():
                if cfg["separator"] == replacement:
                    self.config.style = style
                    self.config.add_spaces = cfg["add_spaces"]
                    break

    def separate_tokens(self, tokens: List[str]) -> str:
        """
        Separate the given tokens according to the configured style.

        Args:
            tokens: List of tokens/characters to separate

        Returns:
            Formatted string with separated tokens
        """
        # Handle empty token list
        if not tokens:
            return ""

        # Process tokens according to configuration
        processed_tokens = [
            token.upper() if self.config.capitalize else token
            for token in tokens
        ]

        # Apply separator
        if self.config.style == SeparatorStyle.NONE:
            return "".join(processed_tokens)

        separator = self.config.separator
        if self.config.add_spaces:
            separator = f" {separator} "

        result = separator.join(processed_tokens)

        # Validate result
        if not self.validator.is_valid_separator(separator):
            # Fall back to space separator if the current one is invalid
            separator = " "
            result = " ".join(processed_tokens)

        return result

    @classmethod
    def get_random_separator(cls) -> 'TokenSeparator':
        """Create a TokenSeparator with a random style."""
        styles = [style for style in SeparatorStyle if style != SeparatorStyle.CUSTOM]
        random_style = random.choice(styles)
        config = SeparatorConfig.from_style(random_style)
        return cls(config)

    @classmethod
    def create_custom(cls, separator: str, add_spaces: bool = True, capitalize: bool = False) -> 'TokenSeparator':
        """Create a TokenSeparator with a custom separator."""
        # Initialize validator to check separator
        validator = TokenizerValidator()

        # If custom separator is invalid, get a valid replacement
        if not validator.is_valid_separator(separator):
            separator = validator.suggest_separator_replacement(separator)

        config = SeparatorConfig(
            style=SeparatorStyle.CUSTOM,
            separator=separator,
            add_spaces=add_spaces,
            capitalize=capitalize
        )
        return cls(config)

def get_all_separator_examples(word: str) -> Dict[str, str]:
    """
    Generate examples of a word with all separator styles.

    Args:
        word: The word to separate

    Returns:
        Dictionary mapping style names to separated examples
    """
    examples = {}
    tokens = list(word)

    for style in SeparatorStyle:
        if style == SeparatorStyle.CUSTOM:
            continue

        separator = TokenSeparator(SeparatorConfig.from_style(style))
        examples[style.value] = separator.separate_tokens(tokens)

    return examples
