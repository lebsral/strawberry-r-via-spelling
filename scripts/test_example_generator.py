#!/usr/bin/env python3
"""
Test script for demonstrating the example generator functionality.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.example_generator import ExampleGenerator, TemplateConfig
from src.data.token_separator import SeparatorStyle

def main():
    """Main function to demonstrate example generator functionality."""
    # Set up configuration
    config = TemplateConfig(
        templates_dir=Path("configs/templates"),
        output_dir=Path("data/processed/template_variations")
    )

    # Create example generator
    generator = ExampleGenerator(config)

    # Test words
    words = ["straw", "hello", "python", "world"]

    print("1. Single Example Generation")
    print("-" * 40)
    single_examples = generator.generate_examples(["straw"], num_variations=1, balance_categories=False)
    if single_examples:
        example = single_examples[0]
        print(f"Instruction: {example['instruction']}")
        print(f"Input      : {example['input']}")
        print(f"Output     : {example['output']}")
    else:
        print("No example generated.")
    print()

    print("2. Multiple Examples with Different Separators")
    print("-" * 40)
    for style in SeparatorStyle:
        if style == SeparatorStyle.CUSTOM:
            continue
        # Use a custom fill for separator style
        examples = []
        for template in generator.templates["spelling_first"]["simple"]:
            try:
                ex = generator.fill_template(template, "hello", {"word": "hello"}, "spelling_first", separator_style=style)
                examples.append(ex)
            except Exception as e:
                print(f"Warning: {str(e)}")
                continue
        if examples:
            example = examples[0]
            print(f"{style.value:10}: Instruction: {example['instruction']} | Input: {example['input']} | Output: {example['output']}")
        else:
            print(f"{style.value:10}: No example generated.")
    print()

    print("3. Balanced Category Examples")
    print("-" * 40)
    examples = generator.generate_examples(
        words=["python"],
        num_variations=2,
        balance_categories=True
    )
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Input      : {example['input']}")
        print(f"Output     : {example['output']}")
        print()

    print("4. Multiple Words with Variations")
    print("-" * 40)
    examples = generator.generate_examples(
        words=words,
        num_variations=2,
        balance_categories=False
    )

    # Save examples to file
    output_file = generator.save_examples(examples)
    print(f"Saved {len(examples)} examples to {output_file}")
    print("\nSample of generated examples:")
    for i, example in enumerate(examples[:4], 1):
        print(f"\nExample {i}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Input      : {example['input']}")
        print(f"Output     : {example['output']}")

if __name__ == "__main__":
    main()
