"""
Token Extractor for GPT-2 Vocabulary

This script extracts all multi-character, letter-based tokens from the GPT-2 tokenizer
vocabulary and saves them to a JSON file.
"""

from transformers import GPT2Tokenizer
import json
import re
import os
from typing import List, Dict
from pathlib import Path

def extract_tokens() -> List[Dict[str, any]]:
    """
    Extract multi-character, letter-based tokens from GPT-2's vocabulary.

    Returns:
        List[Dict[str, any]]: List of dictionaries containing token information
    """
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Extract and filter tokens
    filtered_tokens = []
    for token_text, token_id in tokenizer.get_vocab().items():
        # Skip byte tokens (they start with 'Ġ' or are hex numbers)
        if token_text.startswith('Ġ'):
            token_text = token_text[1:]

        # Filter for multi-character letter-based tokens
        if len(token_text) > 1 and re.match(r'^[a-zA-Z]+$', token_text):
            filtered_tokens.append({
                "token": token_text,
                "token_id": token_id,
                "char_length": len(token_text)
            })

    return filtered_tokens

def save_tokens(tokens: List[Dict[str, any]], output_path: str = "data/processed/gpt2_letter_tokens.json") -> None:
    """
    Save the filtered tokens to a JSON file.

    Args:
        tokens (List[Dict[str, any]]): List of token dictionaries to save
        output_path (str): Path where to save the JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to file
    with open(output_path, "w") as f:
        json.dump({"tokens": tokens}, f, indent=2)

def main():
    """Main execution function"""
    print("Starting token extraction from GPT-2 vocabulary...")

    # Extract tokens
    tokens = extract_tokens()
    print(f"Extracted {len(tokens)} multi-character letter-based tokens")

    # Save tokens
    output_path = "data/processed/gpt2_letter_tokens.json"
    save_tokens(tokens, output_path)
    print(f"Saved tokens to {output_path}")

    # Print some statistics
    token_lengths = [t["char_length"] for t in tokens]
    avg_length = sum(token_lengths) / len(token_lengths)
    print(f"Average token length: {avg_length:.2f} characters")
    print(f"Shortest token length: {min(token_lengths)} characters")
    print(f"Longest token length: {max(token_lengths)} characters")

    # Print a few example tokens
    print("\nExample tokens:")
    for token in sorted(tokens, key=lambda x: x["char_length"])[:5]:
        print(f"Token: {token['token']}, Length: {token['char_length']}")

if __name__ == "__main__":
    main()
