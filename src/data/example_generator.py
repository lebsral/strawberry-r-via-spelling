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
import re

from .token_separator import TokenSeparator, SeparatorStyle, SeparatorConfig
from .token_validator import TokenizerValidator
from .validate_alpaca_schema import AlpacaSchemaValidator

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

    def __init__(self, config: TemplateConfig, validator: TokenizerValidator = None):
        """Initialize the example generator with configuration and optional shared validator."""
        self.config = config
        self.validator = validator or TokenizerValidator()
        self.templates = self._load_templates()
        self.token_separator = TokenSeparator(validator=self.validator)

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

    # Canonical instructions for each category
    CANONICAL_INSTRUCTIONS = {
        "spelling_first": "Spell the following word.",
        "word_first": "How do you spell the following word?",
        "structured": "What letters make up the following word?",
        "char_count_question": "How many characters are in the following word?",
        "char_position_question": "What is the nth letter in the following word?"
    }

    @staticmethod
    def extract_template_vars(template: str) -> set:
        return set(re.findall(r"{(.*?)}", template))

    def fill_template(self, template, word, variables, category, separator_style=None):
        # Always provide 'word' and 'letters' if needed
        if 'letters' in self.extract_template_vars(template):
            # Choose separator style if not provided
            if not separator_style:
                separator_style = SeparatorStyle.SPACE
            separator = TokenSeparator(separator_style, validator=self.validator)
            variables['letters'] = separator.separate_tokens(list(word.lower()))
        try:
            input_text = template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for template: {template}")
        instruction = self.CANONICAL_INSTRUCTIONS.get(category, "Follow the instructions.")
        # Output logic
        if category in ["spelling_first", "word_first", "structured"]:
            # Output is always the letter sequence
            if not separator_style:
                separator_style = SeparatorStyle.SPACE
            separator = TokenSeparator(separator_style, validator=self.validator)
            output_field = separator.separate_tokens(list(word.lower()))
        elif category == "char_count_question":
            if 'letter' in variables:
                output_field = str(word.count(variables['letter']))
            else:
                output_field = str(len(word))
        elif category == "char_position_question":
            n = variables.get('n', 1)
            output_field = word[n-1] if 1 <= n <= len(word) else ""
        else:
            output_field = word
        return {
            'instruction': instruction,
            'input': input_text,
            'output': output_field
        }

    def generate_examples(
        self,
        words: List[str],
        num_variations: int = 1,
        balance_categories: bool = False
    ) -> List[Dict[str, str]]:
        """
        Generate multiple Alpaca-format examples for the given words, robustly handling template variables.
        For spelling/structured categories, generate all combinations of input template and separator style.
        """
        examples = []
        categories = list(self.templates.keys())
        separator_styles = [
            SeparatorStyle.SPACE,
            SeparatorStyle.COMMA,
            SeparatorStyle.DASH,
            SeparatorStyle.PERIOD,
            SeparatorStyle.ARROW
        ]
        for word in words:
            if balance_categories:
                for category, styles in self.templates.items():
                    for style, templates in styles.items():
                        for template in templates:
                            vars_needed = self.extract_template_vars(template)
                            # For spelling/structured, generate all separator styles
                            if category in ["spelling_first", "word_first", "structured"] and vars_needed == {'word'}:
                                for sep_style in separator_styles:
                                    try:
                                        ex = self.fill_template(template, word, {'word': word}, category, separator_style=sep_style)
                                        examples.append(ex)
                                    except Exception as e:
                                        print(f"Warning: {str(e)}")
                                        continue
                            # Only {word} (non-spelling)
                            elif vars_needed == {'word'}:
                                try:
                                    ex = self.fill_template(template, word, {'word': word}, category)
                                    examples.append(ex)
                                except Exception as e:
                                    print(f"Warning: {str(e)}")
                                    continue
                            # {word}, {n}, {ordinal_word}
                            elif vars_needed <= {'word', 'n', 'ordinal_word'}:
                                for n in range(1, len(word)+1):
                                    try:
                                        ex = self.fill_template(
                                            template, word,
                                            {'word': word, 'n': n, 'ordinal_word': self._ordinal_word(n)},
                                            category
                                        )
                                        examples.append(ex)
                                    except Exception as e:
                                        print(f"Warning: {str(e)}")
                                        continue
                            # {word}, {letter}
                            elif vars_needed <= {'word', 'letter'}:
                                for letter in set(word):
                                    try:
                                        ex = self.fill_template(
                                            template, word,
                                            {'word': word, 'letter': letter},
                                            category
                                        )
                                        examples.append(ex)
                                    except Exception as e:
                                        print(f"Warning: {str(e)}")
                                        continue
                            # {word}, {letters}
                            elif vars_needed <= {'word', 'letters'}:
                                for sep_style in separator_styles:
                                    try:
                                        ex = self.fill_template(template, word, {'word': word}, category, separator_style=sep_style)
                                        examples.append(ex)
                                    except Exception as e:
                                        print(f"Warning: {str(e)}")
                                        continue
                            else:
                                print(f"Warning: Cannot fill template: {template} (needs {vars_needed})")
            else:
                for _ in range(num_variations):
                    for category, styles in self.templates.items():
                        for style, templates in styles.items():
                            for template in templates:
                                vars_needed = self.extract_template_vars(template)
                                if category in ["spelling_first", "word_first", "structured"] and vars_needed == {'word'}:
                                    for sep_style in separator_styles:
                                        try:
                                            ex = self.fill_template(template, word, {'word': word}, category, separator_style=sep_style)
                                            examples.append(ex)
                                        except Exception as e:
                                            print(f"Warning: {str(e)}")
                                            continue
                                elif vars_needed == {'word'}:
                                    try:
                                        ex = self.fill_template(template, word, {'word': word}, category)
                                        examples.append(ex)
                                    except Exception as e:
                                        print(f"Warning: {str(e)}")
                                        continue
                                elif vars_needed <= {'word', 'n', 'ordinal_word'}:
                                    n = 1
                                    try:
                                        ex = self.fill_template(
                                            template, word,
                                            {'word': word, 'n': n, 'ordinal_word': self._ordinal_word(n)},
                                            category
                                        )
                                        examples.append(ex)
                                    except Exception as e:
                                        print(f"Warning: {str(e)}")
                                        continue
                                elif vars_needed <= {'word', 'letter'}:
                                    letter = word[0]
                                    try:
                                        ex = self.fill_template(
                                            template, word,
                                            {'word': word, 'letter': letter},
                                            category
                                        )
                                        examples.append(ex)
                                    except Exception as e:
                                        print(f"Warning: {str(e)}")
                                        continue
                                elif vars_needed <= {'word', 'letters'}:
                                    for sep_style in separator_styles:
                                        try:
                                            ex = self.fill_template(template, word, {'word': word}, category, separator_style=sep_style)
                                            examples.append(ex)
                                        except Exception as e:
                                            print(f"Warning: {str(e)}")
                                            continue
                                else:
                                    print(f"Warning: Cannot fill template: {template} (needs {vars_needed})")
        return examples

    def save_examples(self, examples: List[Dict[str, str]], filename: Optional[str] = None) -> Path:
        """
        Save Alpaca-format examples to a JSON file (flat list, not nested).
        Also validates the saved file against the Alpaca schema and prints a warning if any invalid examples are found.
        """
        if filename is None:
            filename = "alpaca_examples.json"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.config.output_dir / filename
        with open(output_file, "w") as f:
            json.dump(examples, f, indent=2)
        # Validate after saving
        validator = AlpacaSchemaValidator()
        report = validator.validate_file(output_file)
        if report["invalid"] > 0:
            print(f"WARNING: {report['invalid']} invalid examples found in {output_file}!")
            for err in report["errors"][:5]:
                print(f"  Example #{err['index']}: {err['errors']}")
        return output_file

    def generate_char_count_example(self, word: str, style: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a character count question example for the given word.
        Returns only Alpaca fields: instruction, input, output.
        Adds a 'word' field for validation (not used in training).
        """
        template = self._get_random_template("char_count_question", "simple")
        input_text = template.format(word=word)
        output = str(len(word))
        instruction = self.CANONICAL_INSTRUCTIONS["char_count_question"]
        return {"instruction": instruction, "input": input_text, "output": output, "word": word}

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

    def generate_count_letter_example(self, word: str) -> Dict[str, str]:
        """Generate an example asking about the number of characters in a word."""
        templates = [
            "How many letters are in the word '{word}'?",
            "What is the length of '{word}'?",
            "Count the number of characters in '{word}'."
        ]
        template = random.choice(templates)
        input_text = template.format(word=word)
        output_text = str(len(word))
        return {
            "input": input_text,
            "output": output_text,
            "template_category": "character_count",
            "template_style": "simple",
            "separator_style": None
        }

    def generate_char_position_examples(self, word: str, positions: List[int] = None) -> List[Dict[str, str]]:
        """Generate examples asking about character positions in a word."""
        if positions is None:
            positions = list(range(1, len(word) + 1))

        templates = [
            "What is the {ordinal} character in '{word}'?",
            "Which letter is at position {position} in '{word}'?",
            "In the word '{word}', what is the {ordinal} letter?"
        ]

        examples = []
        for pos in positions:
            ordinal = self._get_ordinal(pos)
            for template in templates:
                input_text = template.format(word=word, position=pos, ordinal=ordinal)
                output_text = word[pos-1]  # Convert to 0-based index
                examples.append({
                    "input": input_text,
                    "output": output_text,
                    "template_category": "character_position",
                    "template_style": "simple",
                    "separator_style": None
                })
        return examples

    def _get_ordinal(self, n: int) -> str:
        """Convert number to ordinal string (1st, 2nd, 3rd, etc.)"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return f"{n}{suffix}"

    def generate_spelling_example(self, word: str, category: str) -> Dict[str, str]:
        """Generate a spelling example for a word using the specified template category."""
        if category == "spelling_first":
            templates = [
                "How do you spell '{word}'?",
                "What is the correct spelling of '{word}'?",
                "Please spell out the word '{word}'.",
                "Write out the spelling of '{word}'."
            ]
        elif category == "word_first":
            templates = [
                "The word is '{word}'. How is it spelled?",
                "'{word}' - what is its spelling?",
                "Given the word '{word}', write its spelling.",
                "For the word '{word}', provide its spelling."
            ]
        else:  # structured
            templates = [
                "Q: How do you spell '{word}'?\nA: Let me spell that for you:",
                "Q: What is the spelling of '{word}'?\nA: The spelling is:",
                "Q: Can you spell '{word}'?\nA: Here's the spelling:",
                "Q: Write the spelling of '{word}'?\nA: The correct spelling is:"
            ]

        template = random.choice(templates)
        input_text = template.format(word=word)
        output_text = " ".join(word)  # Spell out the word with spaces between letters

        return {
            "input": input_text,
            "output": output_text,
            "template_category": category,
            "template_style": "simple",
            "separator_style": None
        }
