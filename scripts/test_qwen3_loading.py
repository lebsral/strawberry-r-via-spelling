from src.models.qwen3_loader import load_qwen3_tokenizer_only, load_qwen3_model_and_tokenizer, test_inference

if __name__ == "__main__":
    # For data preparation tasks (token extraction, analysis)
    print("Loading Qwen3-4B tokenizer only...")
    tokenizer = load_qwen3_tokenizer_only()
    print("Tokenizer loaded successfully!")

    # Example tokenization
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"Tokenized '{text}' to {tokens}")
    print(f"Decoded back to: '{decoded}'")

    # Only for evaluation/testing (comment out when not needed)
    # print("\nLoading full Qwen3-4B model (only needed for inference)...")
    # model, full_tokenizer, device = load_qwen3_model_and_tokenizer()
    # print("Model loaded successfully!")
    # print("\nTesting basic inference...")
    # test_inference(model, full_tokenizer, device)
