"""
Extract English-only tokens from the Qwen3-4B tokenizer.

- Loads the Qwen3-4B tokenizer using src/models/qwen3_loader.py
- Filters tokens to include only those with strictly alphabetic characters (A-Z, a-z)
- Excludes tokens with numbers, punctuation, or special characters
- Sorts the resulting list alphabetically
- Saves filtered tokens to data/processed/english_tokens_qwen3.txt
- Outputs statistics: total tokens, English tokens, percentage
- Documents methodology and outputs a summary
- Validates by encoding/decoding a sample English sentence
"""
import os
import re
from src.models.qwen3_loader import load_qwen3_tokenizer_only

OUTPUT_PATH = "data/processed/english_tokens_qwen3.txt"
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."

# Regex: strictly alphabetic (A-Z, a-z) only
ENGLISH_TOKEN_PATTERN = re.compile(r'^[A-Za-z]+$')

def is_english_token(token):
    # Token must be strictly alphabetic and not empty
    return bool(ENGLISH_TOKEN_PATTERN.fullmatch(token))

def main():
    print("Loading Qwen3-4B tokenizer...")
    tokenizer = load_qwen3_tokenizer_only()
    vocab = tokenizer.get_vocab()
    print(f"Loaded vocab with {len(vocab)} tokens.")

    english_tokens = []
    for token in vocab:
        if is_english_token(token):
            english_tokens.append(token)

    english_tokens = sorted(set(english_tokens), key=lambda x: x.lower())

    print(f"Filtered {len(english_tokens)} strictly alphabetic English tokens out of {len(vocab)} total tokens.")
    percent = 100 * len(english_tokens) / len(vocab)
    print(f"English token percentage: {percent:.2f}%")

    # Save to file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for token in english_tokens:
            f.write(token + "\n")
    print(f"Saved strictly alphabetic English tokens to {OUTPUT_PATH}")

    # Validation: encode/decode sample text
    print("\nValidation:")
    encoded = tokenizer.encode(SAMPLE_TEXT)
    decoded = tokenizer.decode(encoded)
    print(f"Sample text: {SAMPLE_TEXT}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    print("\nExtraction complete. See script docstring for methodology.")

if __name__ == "__main__":
    main()
