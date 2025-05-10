# PROJECT POLICY: Qwen3-4B must ALWAYS be used in non-thinking mode (enable_thinking=False).
# Any attempt to use thinking mode (enable_thinking=True) is prohibited and will raise an error.
# This applies to all inference, chat template, and tokenizer logic in this project.

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

def apply_qwen3_chat_template_non_thinking(tokenizer, messages, **kwargs):
    """
    Apply the Qwen3 chat template with enable_thinking=False (non-thinking mode only).
    Any attempt to set enable_thinking=True will raise an error.
    """
    if 'enable_thinking' in kwargs and kwargs['enable_thinking']:
        raise ValueError("Project policy violation: enable_thinking=True is not allowed. Only non-thinking mode is supported.")
    # Always enforce non-thinking mode
    kwargs['enable_thinking'] = False
    return tokenizer.apply_chat_template(messages, **kwargs)

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

if __name__ == "__main__":
    tokenizer = load_qwen3_tokenizer_only()
    # This should work (non-thinking mode)
    print(apply_qwen3_chat_template_non_thinking(tokenizer, [{"role": "user", "content": "Hello?"}]))
    # This should raise an error (thinking mode, which is forbidden)
    try:
        apply_qwen3_chat_template_non_thinking(tokenizer, [{"role": "user", "content": "Hello?"}], enable_thinking=True)
    except ValueError as e:
        print("Caught expected error:", e)
