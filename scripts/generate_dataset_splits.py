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

# Config paths
TOKENS_PATH = Path("data/processed/english_tokens.json")
TEMPLATES_DIR = Path("configs/templates")
OUTPUT_DIR = Path("data/processed")

# Multi-token test set generation parameters
WORDS_ALPHA_PATH = Path("data/raw/words_alpha.txt")
MULTI_TOKEN_VAL_PATH = OUTPUT_DIR / "val_char_questions_multi_token.json"
MULTI_TOKEN_TEST_PATH = OUTPUT_DIR / "test_char_questions_multi_token.json"
MULTI_TOKEN_SAMPLE_SIZE = 10000
QWEN_MODEL_NAME = "Qwen/Qwen1.5-4B"

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

    # Split tokens for val/test (no overlap)
    random.shuffle(tokens)
    split = len(tokens) // 2
    val_tokens = tokens[:split]
    test_tokens = tokens[split:]

    # Set up generator
    config = TemplateConfig(templates_dir=TEMPLATES_DIR, output_dir=OUTPUT_DIR)
    generator = ExampleGenerator(config)

    # 1. Training set: spelling examples
    print("Generating training (spelling) examples...")
    spelling_categories = ["spelling_first", "word_first", "structured"]
    train_examples = []
    for word in tokens:
        # For each spelling category, generate one example per word
        for category in spelling_categories:
            train_examples.append(generator.generate_example(word, category=category))
    print(f"Training set: {len(train_examples)} examples.")

    # 2. Validation set: character count, character position, and count_letter questions
    print("Generating validation (character questions) examples...")
    val_examples = []
    for word in val_tokens:
        val_examples.append(generator.generate_char_count_example(word))
        val_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
        val_examples.append(generator.generate_count_letter_example(word))
    print(f"Validation set: {len(val_examples)} examples.")

    # 3. Test set: character count, character position, and count_letter questions
    print("Generating test (character questions) examples...")
    test_examples = []
    for word in test_tokens:
        test_examples.append(generator.generate_char_count_example(word))
        test_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
        test_examples.append(generator.generate_count_letter_example(word))
    print(f"Test set: {len(test_examples)} examples.")

    # Multi-token logic setup (if words_alpha.txt exists)
    multi_token_words = []
    val_multi_tokens = []
    test_multi_tokens = []
    try:
        with open("data/raw/words_alpha.txt") as f:
            all_words = [w.strip() for w in f if w.strip()]
        # Tokenize and filter for multi-token words
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1_5-4B")
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

    # Save
    with open(MULTI_TOKEN_VAL_PATH, "w") as f:
        json.dump({"examples": val_multi}, f, indent=2)
    with open(MULTI_TOKEN_TEST_PATH, "w") as f:
        json.dump({"examples": test_multi}, f, indent=2)
    print(f"Saved multi-token validation set: {MULTI_TOKEN_VAL_PATH} ({len(val_multi)} examples)")
    print(f"Saved multi-token test set: {MULTI_TOKEN_TEST_PATH} ({len(test_multi)} examples)")

if __name__ == "__main__":
    main()
