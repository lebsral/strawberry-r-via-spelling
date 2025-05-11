#!/usr/bin/env python3
"""
Test script for validating tokenizer compatibility changes.

This script tests the token validation system and its integration with
templates and example generation.
"""

import sys
import os
from pathlib import Path
import json

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.token_validator import TokenizerValidator
from src.data.token_separator import TokenSeparator, SeparatorStyle
from src.data.example_generator import ExampleGenerator, TemplateConfig

def test_token_validator():
    """Test the TokenizerValidator functionality."""
    print("\n=== Testing TokenizerValidator ===")
    validator = TokenizerValidator()

    # Test separator validation
    print("\nTesting separator validation:")
    test_separators = [
        " ", ",", "-", ".", "->",  # Should be valid
        "...", "=>", "|", ";"      # Should be invalid
    ]

    for sep in test_separators:
        valid = validator.is_valid_separator(sep)
        replacement = validator.suggest_separator_replacement(sep) if not valid else sep
        print(f"Separator: {sep:4} | Valid: {valid:5} | Replacement: {replacement}")

    # Test template validation
    print("\nTesting template validation:")
    test_templates = [
        "The letters {letters} spell '{word}'",  # Should be valid
        "Let's spell {word}: {letters}",         # Should be valid
        "✨ {letters} ✨ spells {word}",         # Should be invalid (emojis)
        "《 {letters} 》makes {word}",           # Should be invalid (CJK brackets)
    ]

    for template in test_templates:
        valid = validator.validate_template(template)
        print(f"Template valid: {valid:5} | {template}")

def test_token_separator():
    """Test the updated TokenSeparator with validation."""
    print("\n=== Testing TokenSeparator ===")

    # Test all separator styles
    print("\nTesting all separator styles:")
    word = "hello"
    tokens = list(word)

    for style in SeparatorStyle:
        separator = TokenSeparator(style)
        result = separator.separate_tokens(tokens)
        print(f"{style.value:10}: {result}")

    # Test custom separators
    print("\nTesting custom separators:")
    test_separators = ["=>", "...", "|"]
    for sep in test_separators:
        separator = TokenSeparator.create_custom(sep)
        result = separator.separate_tokens(tokens)
        print(f"Custom ({sep:4}): {result}")

def test_example_generator():
    """Test the ExampleGenerator with token validation."""
    print("\n=== Testing ExampleGenerator ===")

    # Set up configuration
    config = TemplateConfig(
        templates_dir=Path("configs/templates"),
        output_dir=Path("data/processed/template_variations")
    )

    # Create generator
    generator = ExampleGenerator(config)

    # Test single example generation
    print("\nTesting single example generation:")
    example = generator.generate_example("python")
    print(json.dumps(example, indent=2))

    # Test multiple examples with different separators
    print("\nTesting examples with different separators:")
    for style in SeparatorStyle:
        try:
            example = generator.generate_example("hello", separator_style=style)
            print(f"{style.value:10}: {example['input']}")
        except ValueError as e:
            print(f"{style.value:10}: Error - {str(e)}")

    # Test batch generation with validation
    print("\nTesting batch generation:")
    words = ["apple", "banana", "こんにちは", "python", "world"]  # Include non-English to test filtering
    examples = generator.generate_examples(words, num_variations=2)
    print(f"Generated {len(examples)} valid examples")
    for i, example in enumerate(examples, 1):
        print(f"\nExample {i}:")
        print(f"Input : {example['input']}")
        print(f"Output: {example['output']}")
        print(f"Category: {example['template_category']}")
        print(f"Style: {example['template_style']}")
        print(f"Separator: {example['separator_style']}")

def main():
    """Main function to run all tests."""
    # Run all tests
    test_token_validator()
    test_token_separator()
    test_example_generator()

if __name__ == "__main__":
    main()
