from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_qwen3_tokenizer_only():
    """
    Load only the Qwen3-4B tokenizer for data preparation, token extraction, and analysis tasks.
    Use this for all non-inference workflows.
    """
    model_id = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return tokenizer

def load_qwen3_model_and_tokenizer():
    """
    Load the Qwen3-4B model and tokenizer for inference or evaluation.
    Only use this when you need to run model.generate or similar methods.
    """
    model_id = "Qwen/Qwen3-4B"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "mps" else torch.float32
        )
        model.to(device)
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, device

def test_inference(model, tokenizer, device):
    if model is None or tokenizer is None:
        print("Model or tokenizer not loaded.")
        return
    prompt = "Hello, world!"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")
