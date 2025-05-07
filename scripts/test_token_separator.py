#!/usr/bin/env python3
"""
Test script for demonstrating the token separator functionality.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.token_separator import (
    TokenSeparator,
    SeparatorStyle,
    SeparatorConfig,
    get_all_separator_examples
)

def main():
    """Main function to demonstrate token separator functionality."""
    # Test word
    word = "straw"
    print(f"Testing token separation for word: {word}\n")

    # Get examples with all separator styles
    examples = get_all_separator_examples(word)

    print("Examples with different separator styles:")
    print("-" * 40)
    for style, separated in examples.items():
        print(f"{style:10}: {separated}")
    print()

    # Demonstrate custom separator
    custom_separator = TokenSeparator.create_custom("=>", add_spaces=True, capitalize=True)
    custom_result = custom_separator.separate_tokens(list(word))
    print("Custom separator example:")
    print(f"custom    : {custom_result}")
    print()

    # Demonstrate random separators
    print("Random separator examples:")
    print("-" * 40)
    for i in range(5):
        random_separator = TokenSeparator.get_random_separator()
        result = random_separator.separate_tokens(list(word))
        print(f"random {i+1:2}: {result}")

if __name__ == "__main__":
    main()
