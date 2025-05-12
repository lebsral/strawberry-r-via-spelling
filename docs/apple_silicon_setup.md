# Apple Silicon (Mac M1/M2) Setup Guide

This guide provides step-by-step instructions, troubleshooting, and performance tips for setting up the project on Mac computers with Apple Silicon (M1/M2).

---

## 1. Environment Setup (Recommended)

### a. Install [uv](https://github.com/astral-sh/uv) (fast Python package manager)

```sh
curl -fsSL https://astral.sh/uv/install.sh | bash
```

### b. Create and activate a virtual environment

```sh
uv venv .venv
source .venv/bin/activate
```

### c. Confirm Python architecture is arm64 (not x86_64/Rosetta)

```sh
python -c 'import platform; print(platform.platform())'
# Output should include 'arm64'
```

If you see 'x86_64', you are running under Rosetta. Use a native terminal (not iTerm2 in Rosetta mode) and ensure your Python is arm64.

### d. Install core dependencies (Apple Silicon wheels)

```sh
uv pip install black ruff mypy ipython requests torch transformers datasets wandb dspy lightning matplotlib seaborn pandas jupyter notebook ipywidgets
```

- **Do NOT install Unsloth or xformers locally.**
- `uv` will fetch Apple Silicon-optimized wheels for torch and transformers.

### e. Install Ollama for local quantized inference

```sh
uv pip install ollama
```

### f. Freeze installed packages

```sh
uv pip freeze > requirements.txt
```

---

## 2. Performance Optimization Tips

- Always use arm64 Python and wheels (never run under Rosetta).
- Use `uv` for faster, reproducible installs.
- Torch will use MPS (Apple Metal) for acceleration. To check:

  ```sh
  python -c 'import torch; print(torch.backends.mps.is_available())'
  # Should print True
  ```

- For best performance, keep macOS and Xcode command line tools up to date.

---

## 3. Troubleshooting Common Issues

| Problem | Solution |
|---------|----------|
| torch/transformers install fails | Ensure you are using arm64 Python and latest pip/uv. Try `brew install libomp` if you see OpenMP errors. |
| Python shows x86_64 | Open a new terminal (not Rosetta), reinstall Python via Homebrew: `brew install python` |
| Jupyter/Matplotlib backend errors | Try `pip install matplotlib ipympl` and use `%matplotlib widget` in notebooks. |
| Ollama import fails | Ensure you installed with `uv pip install ollama` and are not in a cloud environment. |
| CUDA errors | Ignore on Mac; CUDA is not supported. Use MPS (Metal) instead. |
| Homebrew not found | Install from <https://brew.sh> |

---

## 4. Verification Steps

### a. Test script for torch, transformers, and Ollama

Create a file `verify_mac_setup.py`:

```python
import platform
import torch
import transformers
try:
    import ollama
    ollama_ok = True
except ImportError:
    ollama_ok = False

print("Python platform:", platform.platform())
print("Torch version:", torch.__version__)
print("Torch MPS available:", torch.backends.mps.is_available())
print("Transformers version:", transformers.__version__)
print("Ollama import:", ollama_ok)
```

Run:

```sh
python verify_mac_setup.py
```

- Output should show arm64, torch/transformers versions, MPS available: True, and Ollama import: True.

### b. Simple inference test (optional)

```sh
ollama run qwen3-4b:quantized --input data/processed/tokens.json
```

---

## 5. Version Compatibility Notes

- torch >=2.0.0 and transformers >=4.51.0 are recommended for Apple Silicon.
- Use the latest uv and pip for best results.
- Ollama Python API should match your local Ollama install version.

---

## 6. Additional Resources

- [Apple Silicon Python wheels](https://github.com/conda-forge/miniforge)
- [PyTorch MPS backend](https://pytorch.org/docs/stable/notes/mps.html)
- [Ollama documentation](https://github.com/jmorganca/ollama)
- [uv documentation](https://github.com/astral-sh/uv)

---

**If you encounter issues not listed here, please open an issue or check the README for project-specific troubleshooting.**
