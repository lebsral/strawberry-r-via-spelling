"""
Example Generator Implementation

This module provides functionality to generate training examples using templates
and token separation strategies. It supports multiple template categories, styles,
and separator formats to create diverse training data.

Key Features:
- Template-based example generation
- Multiple separator styles (space, comma, dash, etc.)
- Category and style balancing
- Batch example generation
- JSON output formatting

Classes:
    TemplateConfig: Configuration for template loading and example generation
    ExampleGenerator: Main class for generating and saving examples
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

from .token_separator import TokenSeparator, SeparatorStyle, SeparatorConfig

@dataclass
class TemplateConfig:
    """Configuration for template loading and example generation.

    Attributes:
        templates_dir: Path to directory containing template configuration files
        output_dir: Path to directory where generated examples will be saved
    """
    templates_dir: Path
    output_dir: Path
    categories_file: str = "categories.json"

    @property
    def categories_path(self) -> Path:
        """Get the path to the template categories file.

        Returns:
            Path to categories.json file in templates directory
        """
        return self.templates_dir / self.categories_file

class ExampleGenerator:
    """Generates training examples using templates and token separators.

    This class handles loading templates, selecting appropriate formats,
    and generating examples with various styles and separators.

    Attributes:
        config: TemplateConfig instance with paths
        templates: Dictionary of loaded templates by category and style
        token_separator: TokenSeparator instance for letter formatting
    """

    def __init__(self, config: TemplateConfig):
        """Initialize the example generator with configuration."""
        self.config = config
        self.templates = self._load_templates()
        self.token_separator = TokenSeparator()

    def _load_templates(self) -> Dict:
        """Load templates from the categories JSON file."""
        try:
            with open(self.config.categories_path) as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Templates file not found: {self.config.categories_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in templates file: {self.config.categories_path}")

    def _get_random_template(self, category: str, style: Optional[str] = None) -> str:
        """Get a random template from the specified category and style.

        Args:
            category: Template category (e.g., 'spelling_first')
            style: Template style (e.g., 'simple', 'playful'). If None, a random style is chosen.

        Returns:
            A randomly selected template string
        """
        if category not in self.templates:
            raise ValueError(f"Invalid category: {category}")

        # Get available styles for the category
        styles = list(self.templates[category].keys())
        if not styles:
            raise ValueError(f"No styles found for category: {category}")

        # Select style
        selected_style = style if style in styles else random.choice(styles)

        # Get templates for the selected style
        templates = self.templates[category][selected_style]
        if not templates:
            raise ValueError(f"No templates found for style: {selected_style}")

        return random.choice(templates)

    def _detect_template_format(self, template: str) -> Dict[str, str]:
        """
        Detect the format of a template and return appropriate variable mappings.

        Args:
            template: The template string to analyze

        Returns:
            Dictionary mapping template variables to actual values
        """
        # Check for structured formats
        if "\n" in template:  # Token-based format
            if "target:" in template:
                return {"word_var": "word", "letters_var": "letters"}
            elif "input:" in template:
                return {"word_var": "word", "letters_var": "letters"}
            elif "text:" in template:
                return {"word_var": "word", "letters_var": "letters"}
            elif "prompt:" in template:
                return {"word_var": "word", "letters_var": "letters"}
            return {"word_var": "word", "letters_var": "letters"}

        # Check for JSON-like format
        if template.startswith("{\"") and template.endswith("\"}"):
            if "\"target\":" in template:
                return {"word_var": "word", "letters_var": "letters"}
            elif "\"input\":" in template:
                return {"word_var": "word", "letters_var": "letters"}
            elif "\"text\":" in template:
                return {"word_var": "word", "letters_var": "letters"}
            elif "\"prompt\":" in template:
                return {"word_var": "word", "letters_var": "letters"}
            return {"word_var": "word", "letters_var": "letters"}

        # Default format - all templates use {word} and {letters}
        return {"word_var": "word", "letters_var": "letters"}

    def generate_example(
        self,
        word: str,
        category: Optional[str] = None,
        style: Optional[str] = None,
        separator_style: Optional[SeparatorStyle] = None
    ) -> Dict[str, str]:
        """
        Generate a single training example using the specified parameters.

        Args:
            word: The word to generate an example for
            category: Optional template category to use
            style: Optional style within the category to use
            separator_style: Optional separator style to use

        Returns:
            Dictionary containing the generated example and metadata
        """
        # Select random category if none specified
        if category is None:
            category = random.choice(list(self.templates.keys()))

        # Get template
        template = self._get_random_template(category, style)

        # Generate separated letters
        if separator_style is None:
            self.token_separator = TokenSeparator.get_random_separator()
        else:
            self.token_separator = TokenSeparator(SeparatorConfig.from_style(separator_style))

        # Convert word to list of letters and separate them
        letters = list(word.lower())  # Convert to lowercase for consistency
        separated_letters = self.token_separator.separate_tokens(letters)

        # Handle JSON-like templates specially
        if template.startswith("{\"") and template.endswith("\"}"):
            # Create a proper JSON object
            template_data = {
                "word": word,
                "letters": separated_letters
            }
            formatted_text = json.dumps(template_data)
        else:
            # Format normal template
            format_kwargs = {
                "word": word,
                "letters": separated_letters
            }
            formatted_text = template.format(**format_kwargs)

        # Return example with metadata
        return {
            "input": formatted_text,
            "output": word,
            "template_category": category,
            "template_style": style,
            "separator_style": self.token_separator.config.style.value
        }

    def generate_examples(
        self,
        words: List[str],
        num_variations: int = 3,
        balance_categories: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate multiple examples for a list of words.

        Args:
            words: List of words to generate examples for
            num_variations: Number of variations to generate per word
            balance_categories: Whether to balance examples across template categories

        Returns:
            List of generated examples
        """
        examples = []

        # Get available categories
        categories = list(self.templates.keys())

        for word in words:
            if balance_categories:
                # Ensure each category is used at least once if possible
                num_categories = min(len(categories), num_variations)
                selected_categories = random.sample(categories, num_categories)
                remaining = num_variations - num_categories

                # Generate examples with balanced categories
                for category in selected_categories:
                    example = self.generate_example(word, category=category)
                    examples.append(example)

                # Generate remaining examples with random categories
                for _ in range(remaining):
                    example = self.generate_example(word)
                    examples.append(example)
            else:
                # Generate examples with random categories
                for _ in range(num_variations):
                    example = self.generate_example(word)
                    examples.append(example)

        return examples

    def save_examples(self, examples: List[Dict[str, str]], filename: Optional[str] = None) -> Path:
        """
        Save generated examples to a JSON file.

        Args:
            examples: List of examples to save
            filename: Optional filename to save to (default: template_examples.json)

        Returns:
            Path to the saved file
        """
        if filename is None:
            filename = "template_examples.json"

        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.config.output_dir / filename

        # Save examples
        with open(output_file, "w") as f:
            json.dump({"examples": examples}, f, indent=2)

        return output_file
