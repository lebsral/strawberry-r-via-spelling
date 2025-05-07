#!/usr/bin/env python3
"""
Test script to generate multiple spelling examples with different words.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.example_generator import ExampleGenerator, TemplateConfig

def main():
    # Configure paths
    config = TemplateConfig(
        templates_dir=Path("configs/templates"),
        output_dir=Path("data/processed/template_variations")
    )

    # Initialize generator
    generator = ExampleGenerator(config)

    # Test words
    words = [
        "apple", "banana", "cherry", "dolphin", "elephant",
        "flower", "giraffe", "honey", "island", "jungle",
        "koala", "lemon", "mango", "narwhal", "octopus",
        "penguin", "quail", "rabbit", "sunset", "turtle"
    ]

    # Generate examples
    examples = generator.generate_examples(
        words=words,
        num_variations=1,  # One variation per word for clarity
        balance_categories=True  # Use different template categories
    )

    # Print examples nicely
    print("\nGenerated Examples:\n")
    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Input:  {example['input']}")
        print(f"Output: {example['output']}")
        print(f"Category: {example['template_category']}")
        print(f"Style: {example['template_style']}")
        print(f"Separator: {example['separator_style']}")
        print("-" * 80)
        print()

    # Save examples
    output_file = generator.save_examples(examples, "diverse_examples.json")
    print(f"\nExamples saved to: {output_file}")

if __name__ == "__main__":
    main()
