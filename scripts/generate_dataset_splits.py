#!/usr/bin/env python3
"""
Script to generate dataset splits for Qwen3-4B English-only token subset:
- Training: spelling examples
- Validation: character count and character position questions
- Test: character count and character position questions (no overlap with validation)
"""

import json
import random
from pathlib import Path
from src.data.example_generator import ExampleGenerator, TemplateConfig
from transformers import AutoTokenizer
from src.data.token_validator import TokenizerValidator

# Config paths
TOKENS_PATH = Path("data/processed/english_tokens.json")
TEMPLATES_DIR = Path("configs/templates")
OUTPUT_DIR = Path("data/processed")

# Multi-token test set generation parameters
WORDS_ALPHA_PATH = Path("data/raw/words_alpha.txt")
MULTI_TOKEN_VAL_PATH = OUTPUT_DIR / "val_char_questions_multi_token.json"
MULTI_TOKEN_TEST_PATH = OUTPUT_DIR / "test_char_questions_multi_token.json"
MULTI_TOKEN_SAMPLE_SIZE = 10000
QWEN_MODEL_NAME = "unsloth/Qwen3-4B"  # Use this for data generation and fine-tuning with Unsloth
# For inference/deployment with llama.cpp, Ollama, etc., use the GGUF model: https://huggingface.co/unsloth/Qwen3-4B-GGUF

# Load tokens
def load_tokens(words_alpha_set=None):
    with open(TOKENS_PATH) as f:
        data = json.load(f)
    tokens = [t.lower() for t in data["tokens"] if isinstance(t, str) and t]
    if words_alpha_set is not None:
        tokens = [t for t in tokens if t in words_alpha_set]
    return tokens

# Helper: load words_alpha.txt
def load_words_alpha():
    with open(WORDS_ALPHA_PATH) as f:
        return [line.strip().lower() for line in f if line.strip()]

# Helper: filter for multi-token words
def filter_multi_token_words(words, tokenizer):
    filtered = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        if len(tokens) >= 2 and any(len(t.replace("▁", "")) >= 2 for t in tokens):
            filtered.append(word)
    return filtered

def main():
    # Load and lowercase words_alpha.txt
    words_alpha = load_words_alpha()
    words_alpha_set = set(words_alpha)

    # Filter tokens to only those in words_alpha.txt (all lowercased)
    tokens = load_tokens(words_alpha_set=words_alpha_set)
    print(f"Loaded {len(tokens)} tokens after filtering with words_alpha.txt.")
    random.seed(42)

    # LIMIT FOR QUICK TESTING: Only use first 20 tokens
    tokens = tokens[:20]
    print(f"Limiting to {len(tokens)} tokens for quick test run.")

    # Set up validator for English-only subset
    shared_validator = TokenizerValidator()

    # Filter tokens to only those in the English-only subset
    tokens = [t for t in tokens if shared_validator.english_tokens and t in shared_validator.english_tokens]
    print(f"Filtered to {len(tokens)} tokens in the English-only subset.")

    # Split tokens for val/test (no overlap)
    random.shuffle(tokens)
    split = len(tokens) // 2
    val_tokens = tokens[:split]
    test_tokens = tokens[split:]

    # Filter val/test tokens for English-only subset (redundant but safe)
    val_tokens = [t for t in val_tokens if t in shared_validator.english_tokens]
    test_tokens = [t for t in test_tokens if t in shared_validator.english_tokens]

    # Set up generator
    config = TemplateConfig(templates_dir=TEMPLATES_DIR, output_dir=OUTPUT_DIR)
    generator = ExampleGenerator(config, validator=shared_validator)

    # 1. Training set: spelling examples
    print("Generating training (spelling) examples...")
    spelling_categories = ["spelling_first", "word_first", "structured"]
    train_examples = []
    for i, word in enumerate(tokens):
        for category in spelling_categories:
            if i < 10 or (i + 1) % 1000 == 0:
                print(f"Generating example for word: {word}, category: {category} (index {i})")
            # Use generate_examples to get a list of examples for this word/category
            exs = generator.generate_examples([word], num_variations=1, balance_categories=False)
            train_examples.extend(exs)
            if i < 10 or (i + 1) % 1000 == 0:
                print(f"Finished example for word: {word}, category: {category} (index {i})")
        if (i + 1) % 1000 == 0 or (i + 1) == len(tokens):
            print(f"  Progress: {i + 1}/{len(tokens)} words processed ({(i + 1) * len(spelling_categories)} examples)...")
    print(f"Training set: {len(train_examples)} examples.")

    # 2. Validation set: character count, character position, and count_letter questions
    print("Generating validation (character questions) examples...")
    val_examples = []
    for i, word in enumerate(val_tokens):
        val_examples.append(generator.generate_char_count_example(word))
        val_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
        val_examples.append(generator.generate_count_letter_example(word))
        if (i + 1) % 1000 == 0 or (i + 1) == len(val_tokens):
            print(f"  Validation progress: {i + 1}/{len(val_tokens)} words processed...")
    print(f"Validation set: {len(val_examples)} examples.")

    # 3. Test set: character count, character position, and count_letter questions
    print("Generating test (character questions) examples...")
    test_examples = []
    for i, word in enumerate(test_tokens):
        test_examples.append(generator.generate_char_count_example(word))
        test_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
        test_examples.append(generator.generate_count_letter_example(word))
        if (i + 1) % 1000 == 0 or (i + 1) == len(test_tokens):
            print(f"  Test progress: {i + 1}/{len(test_tokens)} words processed...")
    print(f"Test set: {len(test_examples)} examples.")

    # Multi-token logic setup (if words_alpha.txt exists)
    multi_token_words = []
    val_multi_tokens = []
    test_multi_tokens = []
    try:
        with open("data/raw/words_alpha.txt") as f:
            all_words = [w.strip() for w in f if w.strip()]
        # Tokenize and filter for multi-token words
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
        for word in all_words:
            tokens = tokenizer.tokenize(word)
            if len(tokens) >= 2 and any(len(t) >= 2 for t in tokens):
                multi_token_words.append(word)
        random.shuffle(multi_token_words)
        val_multi_tokens = multi_token_words[:MULTI_TOKEN_SAMPLE_SIZE]
        test_multi_tokens = multi_token_words[MULTI_TOKEN_SAMPLE_SIZE:2*MULTI_TOKEN_SAMPLE_SIZE]
    except Exception as e:
        print(f"Skipping multi-token split generation: {e}")
        multi_token_words = []
        val_multi_tokens = []
        test_multi_tokens = []

    # 4. Multi-token validation/test splits (if applicable)
    if multi_token_words and val_multi_tokens and test_multi_tokens:
        print("Generating multi-token validation/test examples...")
        val_multi_examples = []
        test_multi_examples = []
        for word in val_multi_tokens:
            val_multi_examples.append(generator.generate_char_count_example(word))
            val_multi_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
            val_multi_examples.append(generator.generate_count_letter_example(word))
        for word in test_multi_tokens:
            test_multi_examples.append(generator.generate_char_count_example(word))
            test_multi_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
            test_multi_examples.append(generator.generate_count_letter_example(word))
        print(f"Multi-token validation set: {len(val_multi_examples)} examples.")
        print(f"Multi-token test set: {len(test_multi_examples)} examples.")

        # Save multi-token splits
        with open(OUTPUT_DIR / "val_char_questions_multi_token.json", "w") as f:
            json.dump({"examples": val_multi_examples}, f, indent=2)
        with open(OUTPUT_DIR / "test_char_questions_multi_token.json", "w") as f:
            json.dump({"examples": test_multi_examples}, f, indent=2)

    # 5. Save splits
    with open(OUTPUT_DIR / "train_spelling.json", "w") as f:
        json.dump({"examples": train_examples}, f, indent=2)
    with open(OUTPUT_DIR / "val_char_questions.json", "w") as f:
        json.dump({"examples": val_examples}, f, indent=2)
    with open(OUTPUT_DIR / "test_char_questions.json", "w") as f:
        json.dump({"examples": test_examples}, f, indent=2)
    print("Saved all splits.")

    # 6. Print summary
    print("Summary:")
    print(f"  Training: {len(train_examples)} examples, {len(tokens)} unique tokens")
    print(f"  Validation: {len(val_examples)} examples, {len(val_tokens)} unique tokens")
    print(f"  Test: {len(test_examples)} examples, {len(test_tokens)} unique tokens")

    # --- Multi-token splits ---
    print("\nGenerating multi-token validation/test splits from words_alpha.txt ...")
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
    all_words = load_words_alpha()
    multi_token_words = filter_multi_token_words(all_words, tokenizer)
    print(f"Found {len(multi_token_words)} multi-token words.")
    random.shuffle(multi_token_words)
    val_multi = multi_token_words[:MULTI_TOKEN_SAMPLE_SIZE]
    test_multi = multi_token_words[MULTI_TOKEN_SAMPLE_SIZE:2*MULTI_TOKEN_SAMPLE_SIZE]

    # Generate Alpaca-format examples for multi-token splits
    # Filter val_multi and test_multi to only include words in the English-only token set
    val_multi = [w for w in val_multi if w in shared_validator.english_tokens]
    test_multi = [w for w in test_multi if w in shared_validator.english_tokens]
    val_multi_examples = []
    test_multi_examples = []
    for word in val_multi:
        val_multi_examples.append(generator.generate_char_count_example(word))
        val_multi_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
        val_multi_examples.append(generator.generate_count_letter_example(word))
    for word in test_multi:
        test_multi_examples.append(generator.generate_char_count_example(word))
        test_multi_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
        test_multi_examples.append(generator.generate_count_letter_example(word))

    # Save
    with open(MULTI_TOKEN_VAL_PATH, "w") as f:
        json.dump({"examples": val_multi_examples}, f, indent=2)
    with open(MULTI_TOKEN_TEST_PATH, "w") as f:
        json.dump({"examples": test_multi_examples}, f, indent=2)
    print(f"Saved multi-token validation set: {MULTI_TOKEN_VAL_PATH} ({len(val_multi_examples)} examples)")
    print(f"Saved multi-token test set: {MULTI_TOKEN_TEST_PATH} ({len(test_multi_examples)} examples)")

if __name__ == "__main__":
    main()
