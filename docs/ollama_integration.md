# Ollama Integration for Local Quantized Inference (Mac/Apple Silicon)

This guide explains how to install, configure, and use Ollama for local quantized inference with Qwen3-4B on Mac M1/M2.

---

## 1. Installation

### a. Install the Ollama Python package
```sh
uv pip install ollama
```
- Only install Ollama locally on Mac/Apple Silicon. Do **not** install in cloud environments.

### b. System requirements
- Mac M1/M2 (Apple Silicon)
- Python 3.9+
- Sufficient RAM (8GB+ recommended for quantized models)
- Disk space for model weights (Qwen3-4B quantized: several GB)

---

## 2. Configuration for This Project
- No special environment variables are required for local use.
- By default, Ollama will look for models in its standard directory. You can specify a custom model path if needed (see [Ollama docs](https://github.com/jmorganca/ollama)).
- Ensure your Python environment is arm64 (see [apple_silicon_setup.md](apple_silicon_setup.md)).

---

## 3. Example Commands

### a. Run quantized inference with Qwen3-4B
```sh
ollama run qwen3-4b:quantized --input data/processed/tokens.json
```
- Replace `data/processed/tokens.json` with your actual input file.

### b. Use the Ollama Python API
```python
import ollama
response = ollama.run(
    model="qwen3-4b:quantized",
    input_path="data/processed/tokens.json"
)
print(response)
```

---

## 4. Performance Expectations & Resource Requirements
- Quantized inference is optimized for Apple Silicon CPUs/NPUs.
- Expect inference to be much faster than full-precision models, but still slower than cloud GPU.
- RAM usage: 6–8GB typical for Qwen3-4B quantized.
- Disk usage: Model weights may require 4–8GB.
- For best results, close other memory-intensive apps during inference.

---

## 5. Troubleshooting

| Problem | Solution |
|---------|----------|
| `ollama` import fails | Ensure you installed with `uv pip install ollama` in your active virtualenv. |
| Model not found | Download the quantized Qwen3-4B model using the Ollama CLI or specify the correct path. |
| Python architecture is x86_64 | See [apple_silicon_setup.md](apple_silicon_setup.md) to ensure you are using arm64 Python. |
| High RAM usage or slow inference | Close other apps, ensure you are using quantized models, and check system monitor. |
| CUDA errors | Ignore on Mac; Ollama uses CPU/Apple NPU, not CUDA. |

---

## 6. Verification Steps

### a. Import and version check
```python
import ollama
print(ollama.__version__)
```

### b. Run a test inference
```sh
ollama run qwen3-4b:quantized --input data/processed/tokens.json
```
- Output should show model loading and inference results.

---

## 7. Related Documentation
- [Apple Silicon Setup Guide](apple_silicon_setup.md)
- [README: Environment Setup](../README.md#1-environment-setup-macapple-silicon)
- [Ollama official documentation](https://github.com/jmorganca/ollama)

---

**If you encounter issues not listed here, please check the Apple Silicon guide or open an issue.** 
