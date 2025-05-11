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
- Qwen3-4B tokenizer compatibility validation

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
from .token_validator import TokenizerValidator

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
        validator: TokenizerValidator for compatibility checks
    """

    def __init__(self, config: TemplateConfig):
        """Initialize the example generator with configuration."""
        self.config = config
        self.validator = TokenizerValidator()
        self.templates = self._load_templates()
        self.token_separator = TokenSeparator()

    def _load_templates(self) -> Dict:
        """Load and validate templates from the categories JSON file."""
        try:
            with open(self.config.categories_path) as f:
                templates = json.load(f)

            # Validate all templates
            valid_templates = {}
            for category, styles in templates.items():
                valid_styles = {}
                for style, template_list in styles.items():
                    valid_list = []
                    for template in template_list:
                        try:
                            # Try to validate with sample data
                            if self.validator.validate_template(template):
                                valid_list.append(template)
                        except (KeyError, ValueError) as e:
                            print(f"Warning: Invalid template format: {template}")
                            continue

                    if valid_list:  # Only keep styles with valid templates
                        valid_styles[style] = valid_list
                if valid_styles:  # Only keep categories with valid styles
                    valid_templates[category] = valid_styles

            if not valid_templates:
                raise ValueError("No valid templates found after tokenizer validation")

            return valid_templates

        except FileNotFoundError:
            # For testing, create a minimal template set
            print("Warning: Templates file not found. Using minimal test templates.")
            return {
                "spelling_first": {
                    "simple": [
                        "The letters {letters} spell '{word}'",
                        "Let's spell {word}: {letters}"
                    ]
                }
            }
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

        # Only apply separator logic for spelling categories
        spelling_categories = ["spelling_first", "word_first", "structured"]
        if category in spelling_categories:
            allowed_styles = [
                SeparatorStyle.SPACE,
                SeparatorStyle.COMMA,
                SeparatorStyle.DASH,
                SeparatorStyle.PERIOD,
                SeparatorStyle.ARROW
            ]
            style_choice = separator_style or random.choice(allowed_styles)

            # Create separator with validation
            separator = TokenSeparator(style_choice)
            letters = separator.separate_tokens(list(word.lower()))
            sep_style = separator.config.style.value
        else:
            # For non-spelling categories, fallback to default separator logic
            separator = TokenSeparator(SeparatorStyle.SPACE)
            letters = separator.separate_tokens(list(word.lower()))
            sep_style = "space"

        # Format the example
        format_kwargs = {
            "word": word,
            "letters": letters
        }

        # Handle JSON-like templates specially
        if template.startswith("{\"") and template.endswith("\"}"):
            template_data = {
                "word": word,
                "letters": letters
            }
            formatted_text = json.dumps(template_data)
        else:
            formatted_text = template.format(**format_kwargs)

        # Create example
        example = {
            "input": formatted_text,
            "output": word,
            "template_category": category,
            "template_style": style or "simple",
            "separator_style": sep_style
        }

        # Validate the generated example
        if not self.validator.validate_example(example):
            # If invalid, try again with space separator
            separator = TokenSeparator(SeparatorStyle.SPACE)
            letters = separator.separate_tokens(list(word.lower()))

            # Reformat with space separator
            format_kwargs["letters"] = letters
            if template.startswith("{\"") and template.endswith("\"}"):
                template_data["letters"] = letters
                formatted_text = json.dumps(template_data)
            else:
                formatted_text = template.format(**format_kwargs)

            example["input"] = formatted_text
            example["separator_style"] = "space"

            # If still invalid, raise error
            if not self.validator.validate_example(example):
                raise ValueError(
                    f"Unable to generate valid example for word '{word}' "
                    f"with template category '{category}'"
                )

        return example

    def generate_examples(
        self,
        words: List[str],
        num_variations: int = 1,
        balance_categories: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate multiple examples for the given words.

        Args:
            words: List of words to generate examples for
            num_variations: Number of variations to generate per word
            balance_categories: Whether to balance examples across categories

        Returns:
            List of generated examples
        """
        examples = []
        categories = list(self.templates.keys())

        for word in words:
            if balance_categories:
                # Generate one example per category
                for category in categories:
                    try:
                        example = self.generate_example(word, category=category)
                        examples.append(example)
                    except ValueError as e:
                        print(f"Warning: {str(e)}")
                        continue
            else:
                # Generate random variations
                for _ in range(num_variations):
                    try:
                        example = self.generate_example(word)
                        examples.append(example)
                    except ValueError as e:
                        print(f"Warning: {str(e)}")
                        continue

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

    def generate_char_count_example(self, word: str, style: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a character count question example for the given word.
        Args:
            word: The word to generate a question for
            style: Optional style within the category to use
        Returns:
            Dictionary containing the generated example and metadata
        """
        template = self._get_random_template("char_count_question", "simple")
        input_text = template.format(word=word)
        output = str(len(word))
        return {"input": input_text, "output": output, "category": "char_count_question", "subtype": "simple", "word": word}

    def _ordinal(self, n: int) -> str:
        """Return the ordinal string for an integer (e.g., 1 -> '1st')."""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    def _ordinal_word(self, n: int) -> str:
        """Return the spelled-out ordinal word for an integer (e.g., 1 -> 'first')."""
        words = {
            1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
            6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth',
            11: 'eleventh', 12: 'twelfth', 13: 'thirteenth', 14: 'fourteenth', 15: 'fifteenth',
            16: 'sixteenth', 17: 'seventeenth', 18: 'eighteenth', 19: 'nineteenth', 20: 'twentieth'
        }
        return words.get(n, f"{n}th")

    def generate_count_letter_example(self, word: str, style: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a question about counting a specific letter in the word.
        Args:
            word: The word to generate a question for
            style: Optional style within the category to use
        Returns:
            Dictionary containing the generated example and metadata
        """
        # Pick a letter that occurs at least once in the word
        unique_letters = list(set(word))
        letter = random.choice(unique_letters)
        template = self._get_random_template("char_count_question", "count_letter")
        input_text = template.format(word=word, letter=letter)
        output = str(word.count(letter))
        return {"input": input_text, "output": output, "category": "char_count_question", "subtype": "count_letter", "letter": letter, "word": word}

    def generate_char_position_examples(self, word: str, positions: Optional[List[int]] = None, style: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Generate character position question examples for the given word.
        Args:
            word: The word to generate questions for
            positions: List of positions (1-based)
            style: Optional style within the category to use
        Returns:
            List of dictionaries containing generated examples and metadata
        """
        if positions is None:
            positions = list(range(1, len(word) + 1))
        examples = []
        for n in positions:
            template = self._get_random_template("char_position_question", style)
            # Detect which variables are present in the template
            variables = {}
            if "{n}" in template:
                variables["n"] = n
            if "{ordinal}" in template:
                variables["ordinal"] = self._ordinal(n)
            if "{ordinal_word}" in template:
                variables["ordinal_word"] = self._ordinal_word(n)
            input_text = template.format(word=word, **variables)
            output = word[n - 1] if 1 <= n <= len(word) else ""
            examples.append({"input": input_text, "output": output, "category": "char_position_question", "position": n, "word": word})
        return examples
