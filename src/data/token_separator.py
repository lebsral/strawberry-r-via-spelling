"""
Token Separation Strategy Implementation

This module provides various token separation strategies for formatting spelling examples.
It supports different styles of separating characters/tokens for enhanced learning.
"""

from typing import List, Dict, Optional, Union
import random
from dataclasses import dataclass
from enum import Enum

class SeparatorStyle(Enum):
    """Enumeration of available separator styles."""
    NONE = "none"  # No separator
    SPACE = "space"  # Simple space
    COMMA = "comma"  # Comma with space
    DASH = "dash"  # Hyphen
    DOTS = "dots"  # Ellipsis
    ARROW = "arrow"  # Arrow symbol
    CUSTOM = "custom"  # Custom separator

@dataclass
class SeparatorConfig:
    """Configuration for token separation."""
    style: SeparatorStyle
    separator: str
    capitalize: bool = False
    add_spaces: bool = True

    @classmethod
    def from_style(cls, style: SeparatorStyle) -> 'SeparatorConfig':
        """Create a separator configuration from a style."""
        style_configs = {
            SeparatorStyle.NONE: cls(SeparatorStyle.NONE, ""),
            SeparatorStyle.SPACE: cls(SeparatorStyle.SPACE, " "),
            SeparatorStyle.COMMA: cls(SeparatorStyle.COMMA, ",", add_spaces=True),
            SeparatorStyle.DASH: cls(SeparatorStyle.DASH, "-"),
            SeparatorStyle.DOTS: cls(SeparatorStyle.DOTS, "..."),
            SeparatorStyle.ARROW: cls(SeparatorStyle.ARROW, "->", add_spaces=True),
        }
        return style_configs.get(style, cls(SeparatorStyle.SPACE, " "))

class TokenSeparator:
    """Handles the separation of tokens/characters with various styles."""

    def __init__(self, config: Optional[Union[SeparatorConfig, SeparatorStyle]] = None):
        """Initialize the token separator with optional configuration."""
        if isinstance(config, SeparatorStyle):
            self.config = SeparatorConfig.from_style(config)
        else:
            self.config = config or SeparatorConfig.from_style(SeparatorStyle.SPACE)

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

        return separator.join(processed_tokens)

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
