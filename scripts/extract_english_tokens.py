"""
Extract English-only tokens from the Qwen3-4B tokenizer.

- Loads the Qwen3-4B tokenizer using src/models/qwen3_loader.py
- Filters tokens to include only those with strictly alphabetic characters (A-Z, a-z)
- Excludes tokens with numbers, punctuation, or special characters
- Sorts the resulting list alphabetically
- Saves filtered tokens to data/processed/english_tokens.json (JSON, as a dict with a 'tokens' key)
- Outputs statistics: total tokens, English tokens, percentage
- Documents methodology and outputs a summary
- Validates by encoding/decoding a sample English sentence
- The JSON output is the canonical format for downstream scripts and notebooks (see src/data/create_notebook.py)
"""
import os
import re
import json
from src.models.qwen3_loader import load_qwen3_tokenizer_only

OUTPUT_JSON_PATH = "data/processed/english_tokens.json"
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
WORDS_ALPHA_PATH = "data/raw/words_alpha.txt"
MULTI_TOKEN_JSON_PATH = "data/processed/english_multi_tokens.json"

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

    # Save to .json file (canonical for downstream use)
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"tokens": english_tokens}, f, indent=2, ensure_ascii=False)
    print(f"Saved strictly alphabetic English tokens to {OUTPUT_JSON_PATH} (as JSON with 'tokens' key)")

    # Validation: encode/decode sample text
    print("\nValidation:")
    encoded = tokenizer.encode(SAMPLE_TEXT)
    decoded = tokenizer.decode(encoded)
    print(f"Sample text: {SAMPLE_TEXT}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # --- Multi-token extraction ---
    print("\nExtracting multi-token words from words_alpha.txt ...")
    multi_token_words = []
    if os.path.exists(WORDS_ALPHA_PATH):
        with open(WORDS_ALPHA_PATH) as f:
            all_words = [line.strip() for line in f if line.strip()]
        for word in all_words:
            tokens = tokenizer.tokenize(word)
            if len(tokens) >= 2:
                multi_token_words.append(word)
        multi_token_words = sorted(set(multi_token_words), key=lambda x: x.lower())
        print(f"Found {len(multi_token_words)} multi-token words out of {len(all_words)} total words.")
        with open(MULTI_TOKEN_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"tokens": multi_token_words}, f, indent=2, ensure_ascii=False)
        print(f"Saved multi-token words to {MULTI_TOKEN_JSON_PATH} (as JSON with 'tokens' key)")
    else:
        print(f"{WORDS_ALPHA_PATH} not found. Skipping multi-token extraction.")

    print("\nExtraction complete. See script docstring for methodology and output formats.")

if __name__ == "__main__":
    main()
