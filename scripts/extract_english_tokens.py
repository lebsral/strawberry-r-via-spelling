"""
Extract English-only tokens from the Qwen3-4B tokenizer.

- Loads the Qwen3-4B tokenizer using src/models/qwen3_loader.py
- Filters tokens to include only those with strictly alphabetic characters (A-Z, a-z)
- Identifies which tokens are complete English words by checking against words_alpha.txt
- Finds which words tokenize into multiple tokens that are themselves complete words
- Saves results to three files:
  * english_tokens.json: All alphabetic tokens
  * complete_word_tokens.json: Tokens that are themselves complete English words
  * multi_word_tokens.json: Words that tokenize into multiple complete word tokens
"""
import os
import re
import json
from src.models.qwen3_loader import load_qwen3_tokenizer_only

OUTPUT_JSON_PATH = "data/processed/english_tokens.json"
COMPLETE_WORDS_PATH = "data/processed/complete_word_tokens.json"
MULTI_WORD_PATH = "data/processed/multi_word_tokens.json"
SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."
WORDS_ALPHA_PATH = "data/raw/words_alpha.txt"

# Regex: strictly alphabetic (A-Z, a-z) only
ENGLISH_TOKEN_PATTERN = re.compile(r'^[A-Za-z]+$')

def is_english_token(token):
    # Token must be strictly alphabetic and not empty
    return bool(ENGLISH_TOKEN_PATTERN.fullmatch(token))

def find_multi_word_tokens(tokenizer, complete_word_tokens, words_alpha):
    """
    Find words that tokenize into multiple tokens, where each token is itself a complete word.
    For example, if 'sidewalk' tokenizes into ['side', 'walk'] and both 'side' and 'walk' are
    complete word tokens, then 'sidewalk' would be identified.

    Args:
        tokenizer: The Qwen3 tokenizer
        complete_word_tokens: Set of tokens that are complete English words
        words_alpha: Set of valid English words from words_alpha.txt
    """
    # Convert to lowercase for matching
    complete_words_lower = {t.lower() for t in complete_word_tokens}
    multi_word_tokens = []

    # Minimum length for each component to be considered valid
    MIN_TOKEN_LENGTH = 3

    # Common word patterns that form compounds
    COMMON_ENDINGS = {'man', 'men', 'side', 'walk', 'way', 'room', 'time', 'work', 'book', 'house',
                     'land', 'field', 'ship', 'yard', 'wood', 'light', 'line', 'stone', 'town',
                     'water', 'world', 'maker', 'master', 'worker', 'keeper', 'holder', 'writer',
                     'ground', 'ware', 'wear', 'born', 'bound', 'proof', 'like', 'less', 'full'}

    COMMON_BEGINNINGS = {'back', 'book', 'down', 'fire', 'foot', 'hand', 'head', 'home', 'land',
                        'life', 'light', 'moon', 'north', 'out', 'over', 'rail', 'sea', 'ship',
                        'side', 'snow', 'south', 'sun', 'under', 'up', 'water', 'wood', 'work',
                        'air', 'all', 'arm', 'blood', 'brain', 'child', 'day', 'door', 'earth',
                        'eye', 'gold', 'heart', 'high', 'iron', 'law', 'long', 'low', 'main',
                        'man', 'mind', 'night', 'self', 'silver', 'spring', 'steam', 'summer',
                        'war', 'winter'}

    print("\nFinding words that tokenize into multiple complete word tokens...")
    print("This may take a few minutes...")
    print(f"Using {len(complete_word_tokens)} complete word tokens as components")
    print(f"Minimum token length: {MIN_TOKEN_LENGTH} characters")
    print(f"Using {len(COMMON_BEGINNINGS)} common beginnings and {len(COMMON_ENDINGS)} common endings")

    # Process each word in the English dictionary
    for word in words_alpha:
        # Skip short words that can't be meaningful compounds
        if len(word) < 6:  # Minimum reasonable length for a compound
            continue

        # Get the tokens for this word
        tokens = tokenizer.tokenize(word)

        # Skip if it's not a multi-token word
        if len(tokens) < 2:
            continue

        # Clean the tokens (remove any special characters)
        clean_tokens = []
        for token in tokens:
            # Remove any special characters the tokenizer might add
            clean_token = token.strip('▁Ġ')
            if clean_token:
                clean_tokens.append(clean_token)

        # Check if:
        # 1. The word itself exists in words_alpha
        # 2. All component tokens are complete words and meet minimum length
        # 3. All component tokens also exist in words_alpha
        # 4. Either the first token is a common beginning or the last token is a common ending
        if (len(clean_tokens) >= 2 and
            word.lower() in words_alpha and
            all(len(token) >= MIN_TOKEN_LENGTH and
                token.lower() in complete_words_lower and
                token.lower() in words_alpha
                for token in clean_tokens) and
            (clean_tokens[0].lower() in COMMON_BEGINNINGS or
             clean_tokens[-1].lower() in COMMON_ENDINGS)):
            multi_word_tokens.append({
                "word": word,
                "tokens": clean_tokens
            })

            # Print progress occasionally
            if len(multi_word_tokens) % 100 == 0:
                print(f"Found {len(multi_word_tokens)} multi-word tokens so far...")

    # Sort by word
    multi_word_tokens.sort(key=lambda x: x["word"])

    # Show some statistics
    if multi_word_tokens:
        print("\nToken count statistics:")
        token_counts = [len(word["tokens"]) for word in multi_word_tokens]
        avg_tokens = sum(token_counts) / len(token_counts)
        print(f"Average tokens per word: {avg_tokens:.1f}")
        print(f"Token count distribution: {sorted(set(token_counts))}")

        # Show some examples grouped by token count
        print("\nExamples by token count:")
        for count in sorted(set(token_counts)):
            examples = [w for w in multi_word_tokens if len(w["tokens"]) == count][:3]
            if examples:
                print(f"\n{count} tokens:")
                for ex in examples:
                    print(f"  {ex['word']} = {' + '.join(ex['tokens'])}")

    return multi_word_tokens

def main():
    print("Loading Qwen3-4B tokenizer...")
    tokenizer = load_qwen3_tokenizer_only()
    vocab = tokenizer.get_vocab()
    print(f"Loaded vocab with {len(vocab)} tokens.")

    # First get all alphabetic tokens
    english_tokens = []
    for token in vocab:
        if is_english_token(token):
            english_tokens.append(token)

    english_tokens = sorted(set(english_tokens), key=lambda x: x.lower())
    print(f"Filtered {len(english_tokens)} strictly alphabetic English tokens out of {len(vocab)} total tokens.")

    # Save all alphabetic tokens
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"tokens": english_tokens}, f, indent=2, ensure_ascii=False)
    print(f"Saved strictly alphabetic English tokens to {OUTPUT_JSON_PATH}")

    # Validation: encode/decode sample text
    print("\nValidation:")
    encoded = tokenizer.encode(SAMPLE_TEXT)
    decoded = tokenizer.decode(encoded)
    print(f"Sample text: {SAMPLE_TEXT}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Cross-reference with words_alpha.txt to find complete word tokens
    if os.path.exists(WORDS_ALPHA_PATH):
        print("\nIdentifying tokens that are complete English words...")
        with open(WORDS_ALPHA_PATH) as f:
            real_words = {line.strip().lower() for line in f if line.strip()}

        # Find which of our tokens are complete words
        complete_word_tokens = []
        for token in english_tokens:
            if token.lower() in real_words:
                complete_word_tokens.append(token)

        complete_word_tokens = sorted(set(complete_word_tokens), key=lambda x: x.lower())
        print(f"Found {len(complete_word_tokens)} tokens that are complete English words")

        # Save complete word tokens
        with open(COMPLETE_WORDS_PATH, "w", encoding="utf-8") as f:
            json.dump({"words": complete_word_tokens}, f, indent=2, ensure_ascii=False)
        print(f"Saved complete word tokens to {COMPLETE_WORDS_PATH}")

        # Now find words that tokenize into multiple complete word tokens
        multi_word_tokens = find_multi_word_tokens(tokenizer, complete_word_tokens, real_words)
        print(f"\nFound {len(multi_word_tokens)} words that tokenize into multiple complete word tokens.")

        # Save multi-word tokens
        with open(MULTI_WORD_PATH, "w", encoding="utf-8") as f:
            json.dump({"words": multi_word_tokens}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved multi-word tokens to {MULTI_WORD_PATH}")
    else:
        print(f"{WORDS_ALPHA_PATH} not found. Cannot identify complete words.")

    print("\nExtraction complete. See script docstring for methodology and output formats.")

if __name__ == "__main__":
    main()
