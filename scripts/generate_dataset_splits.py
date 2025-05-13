#!/usr/bin/env python3
"""
Script to generate dataset splits for Qwen3-4B English-only token subset:
- Training: spelling examples using component tokens (from component_tokens.json)
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
COMPONENT_TOKENS_PATH = Path("data/processed/component_tokens.json")
TEMPLATES_DIR = Path("configs/templates")
OUTPUT_DIR = Path("data/processed")

# Multi-token test set generation parameters
WORDS_ALPHA_PATH = Path("data/raw/words_alpha.txt")
MULTI_TOKEN_VAL_PATH = OUTPUT_DIR / "val_char_questions_multi_token.json"
MULTI_TOKEN_TEST_PATH = OUTPUT_DIR / "test_char_questions_multi_token.json"
MULTI_TOKEN_SAMPLE_SIZE = 10000
QWEN_MODEL_NAME = "unsloth/Qwen3-4B"  # Use this for data generation and fine-tuning with Unsloth
# For inference/deployment with llama.cpp, Ollama, etc., use the GGUF model: https://huggingface.co/unsloth/Qwen3-4B-GGUF

def load_component_tokens():
    """Load component tokens that are used to form compound words."""
    print("Loading component tokens...")
    with open(COMPONENT_TOKENS_PATH) as f:
        data = json.load(f)
    tokens = data["tokens"]
    usage_stats = data["usage_stats"]["token_frequencies"]
    # Sort tokens by usage frequency (most used first)
    sorted_tokens = sorted(tokens, key=lambda t: usage_stats.get(t, 0), reverse=True)
    print(f"Loaded {len(sorted_tokens)} component tokens.")
    return sorted_tokens, usage_stats

def load_words_alpha():
    with open(WORDS_ALPHA_PATH) as f:
        return [line.strip().lower() for line in f if line.strip()]

def filter_multi_token_words(words, tokenizer):
    filtered = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        if len(tokens) >= 2 and any(len(t.replace("▁", "")) >= 2 for t in tokens):
            filtered.append(word)
    return filtered

def main():
    """Main entry point."""
    # Load component tokens
    print("Loading component tokens...")
    with open(COMPONENT_TOKENS_PATH) as f:
        data = json.load(f)
        component_tokens = data["tokens"]
        token_usage = data["usage"]
    print(f"Loaded {len(component_tokens)} component tokens.")

    # Filter to tokens that are in words_alpha.txt
    words_alpha = load_words_alpha()
    component_tokens = [t for t in component_tokens if t.lower() in words_alpha]
    print(f"Filtered to {len(component_tokens)} component tokens that are in words_alpha.txt")

    # Filter to tokens in English-only subset
    with open(TOKENS_PATH) as f:
        english_tokens = set(json.load(f)["tokens"])
    component_tokens = [t for t in component_tokens if t in english_tokens]
    print(f"Filtered to {len(component_tokens)} component tokens in the English-only subset.")

    # Sort by usage frequency and take top half for training
    sorted_tokens = sorted(component_tokens, key=lambda t: token_usage.get(t, 0), reverse=True)
    train_tokens = sorted_tokens[:len(sorted_tokens)//2]
    val_test_tokens = sorted_tokens[len(sorted_tokens)//2:]
    print(f"Using {len(train_tokens)} most common component tokens for training")

    # Initialize example generator with config
    config = TemplateConfig(templates_dir=TEMPLATES_DIR, output_dir=OUTPUT_DIR)
    generator = ExampleGenerator(config)

    # Generate training examples
    print("Generating training (spelling) examples using component tokens...")
    train_examples = []
    for i, token in enumerate(train_tokens, 1):
        usage = token_usage.get(token, 0)
        # Generate more examples for frequently used tokens
        num_examples = 3 if usage > 400 else (2 if usage > 200 else 1)

        # Generate examples for each template category
        for category in ["spelling_first", "word_first", "structured"]:
            print(f"Generating {num_examples} examples for component: {token} (used in {usage} compounds), category: {category}")
            for _ in range(num_examples):
                example = generator.generate_spelling_example(token, category)
                train_examples.append(example)

        if i % 100 == 0:
            print(f"  Progress: {i}/{len(train_tokens)} components processed...")
    print(f"  Progress: {len(train_tokens)}/{len(train_tokens)} components processed...")
    print(f"Training set: {len(train_examples)} examples generated from {len(train_tokens)} component tokens.")

    # Generate validation examples
    print("Generating validation (character questions) examples...")
    val_examples = []
    for word in val_test_tokens[:len(val_test_tokens)//2]:
        # Character count example
        val_examples.append(generator.generate_count_letter_example(word))
        # Character position example (random position)
        val_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))

    # Generate test examples
    print("Generating test (character questions) examples...")
    test_examples = []
    for word in val_test_tokens[len(val_test_tokens)//2:]:
        # Character count example
        test_examples.append(generator.generate_count_letter_example(word))
        # Character position example (random position)
        test_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))

    # Save the datasets
    print("\nSaving datasets...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "train_spelling.json", "w") as f:
        json.dump({"examples": train_examples}, f, indent=2)
    print(f"Saved {len(train_examples)} training examples")

    with open(OUTPUT_DIR / "val_char.json", "w") as f:
        json.dump({"examples": val_examples}, f, indent=2)
    print(f"Saved {len(val_examples)} validation examples")

    with open(OUTPUT_DIR / "test_char.json", "w") as f:
        json.dump({"examples": test_examples}, f, indent=2)
    print(f"Saved {len(test_examples)} test examples")

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

    # 6. Print summary
    print("Summary:")
    print(f"  Training: {len(train_examples)} examples, {len(train_tokens)} unique component tokens")
    print(f"  Validation: {len(val_examples)} examples, {len(val_test_tokens[:len(val_test_tokens)//2])} unique component tokens")
    print(f"  Test: {len(test_examples)} examples, {len(val_test_tokens[len(val_test_tokens)//2:])} unique component tokens")

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
    val_multi = [w for w in val_multi if w in english_tokens]
    test_multi = [w for w in test_multi if w in english_tokens]
    val_multi_examples = []
    test_multi_examples = []

    print("\nGenerating multi-token validation examples...")
    for i, word in enumerate(val_multi):
        val_multi_examples.append(generator.generate_char_count_example(word))
        val_multi_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
        val_multi_examples.append(generator.generate_count_letter_example(word))
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(val_multi)} words processed...")

    print("\nGenerating multi-token test examples...")
    for i, word in enumerate(test_multi):
        test_multi_examples.append(generator.generate_char_count_example(word))
        test_multi_examples.extend(generator.generate_char_position_examples(word, positions=[random.randint(1, len(word))]))
        test_multi_examples.append(generator.generate_count_letter_example(word))
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(test_multi)} words processed...")

    # Save multi-token splits
    with open(MULTI_TOKEN_VAL_PATH, "w") as f:
        json.dump({"examples": val_multi_examples}, f, indent=2)
    with open(MULTI_TOKEN_TEST_PATH, "w") as f:
        json.dump({"examples": test_multi_examples}, f, indent=2)
    print(f"\nSaved multi-token validation set: {MULTI_TOKEN_VAL_PATH} ({len(val_multi_examples)} examples)")
    print(f"Saved multi-token test set: {MULTI_TOKEN_TEST_PATH} ({len(test_multi_examples)} examples)")

if __name__ == "__main__":
    main()
