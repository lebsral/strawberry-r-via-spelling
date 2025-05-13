"""
Extract the unique component tokens that make up multi-word tokens.

For example, if we have:
- 'sidewalk' = ['side', 'walk']
- 'walkway' = ['walk', 'way']

This script will extract the unique set: ['side', 'walk', 'way']

These are the fundamental building blocks used to create compound words in our tokenizer.
"""
import os
import json
from collections import Counter

# Input/Output paths
MULTI_WORD_PATH = "data/processed/multi_word_tokens.json"
COMPONENT_TOKENS_PATH = "data/processed/component_tokens.json"

def main():
    print("Loading multi-word tokens...")
    with open(MULTI_WORD_PATH) as f:
        data = json.load(f)
        multi_word_tokens = data["words"]

    # Extract all component tokens and count their usage
    component_tokens = set()
    token_usage = Counter()

    print("Extracting component tokens and counting usage...")
    for word_data in multi_word_tokens:
        word = word_data["word"]
        tokens = word_data["tokens"]  # The tokens field contains the component tokens
        for token in tokens:
            component_tokens.add(token)
            token_usage[token] += 1

    # Convert to sorted list by usage frequency
    component_tokens = sorted(component_tokens, key=lambda t: token_usage[t], reverse=True)

    # Save results
    print("\nSaving results...")
    with open(COMPONENT_TOKENS_PATH, "w") as f:
        json.dump({
            "tokens": list(component_tokens),
            "usage": {token: count for token, count in token_usage.items()}
        }, f, indent=2)

    # Print statistics
    print(f"\nFound {len(component_tokens)} unique component tokens")
    print("\nTop 10 most used component tokens:")
    for token in component_tokens[:10]:
        print(f"  {token}: used in {token_usage[token]} compound words")

if __name__ == "__main__":
    main()
