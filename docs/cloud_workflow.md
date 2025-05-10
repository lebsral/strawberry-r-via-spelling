# Cloud Workflow Guide

This guide explains how to set up, run, and manage cloud-based workflows for this project, and how they differ from local (Mac/Apple Silicon) workflows.

---

## 1. When and Why Use Cloud Workflows?
- **Use the cloud for:**
  - Fine-tuning and training with Unsloth and xformers (requires CUDA GPU)
  - Large-scale data processing or evaluation that exceeds local resources
  - Any workflow that requires GPU acceleration or is not supported on Mac/Apple Silicon
- **Use local workflows for:**
  - Data preparation, token extraction, and template generation
  - Local quantized inference with Ollama
  - Development and debugging

---

## 2. Recommended Cloud Platforms
- **Google Colab** (Pro/Pro+ recommended for longer jobs)
- **Lightning AI**
- **Custom cloud VM** (AWS, GCP, Azure, etc. with CUDA-enabled GPU)

---

## 3. Step-by-Step Cloud Setup

### a. Start a new cloud instance or Colab notebook
- Ensure you have access to a CUDA-enabled GPU (NVIDIA)

### b. Install Python and pip (if not pre-installed)
- Most platforms come with Python 3.9+ and pip

### c. Install required dependencies
```sh
pip install torch transformers datasets wandb dspy lightning matplotlib seaborn pandas jupyter notebook ipywidgets unsloth xformers
```
- **Only install `unsloth` and `xformers` in the cloud.**
- Use the latest compatible versions for all packages.

### d. Set up GPU drivers (if needed)
- Colab and Lightning handle this automatically
- For custom VMs, follow your provider's instructions for CUDA and cuDNN

---

## 4. Environment Variables and Credentials
- Copy `.env.example` to `.env` and fill in required API keys (e.g., Hugging Face, W&B)
- Never commit `.env` with secrets to version control
- In Colab, you can set environment variables with:
  ```python
  import os
  os.environ["WANDB_API_KEY"] = "your-key-here"
  ```

---

## 5. Security Considerations
- **Never share or commit API keys**
- Use Colab/Lightning secrets or environment variables for sensitive data
- Be mindful of data privacy when uploading datasets to the cloud

---

## 6. Cost Management Tips
- Use Colab Pro/Pro+ for longer jobs, but shut down idle notebooks to avoid charges
- For custom VMs, choose the smallest GPU instance that meets your needs and stop/delete when not in use
- Monitor GPU and RAM usage to avoid over-provisioning

---

## 7. Example Cloud Workflow Commands

### a. Install dependencies
```sh
pip install torch transformers unsloth xformers
```

### b. Run training or fine-tuning
```sh
python scripts/train.py --model Qwen3-4B --data data/processed/tokens.json --output results/model/
```

### c. Run token extraction or data prep (if needed in cloud)
```sh
python scripts/token_extraction.py --model Qwen3-4B --input data/raw/words.txt --output data/processed/tokens.json
```

---

## 8. Local vs. Cloud Workflow Comparison

| Step                        | Local (Mac/Apple Silicon)         | Cloud (Colab/Lightning/VM)         |
|-----------------------------|-----------------------------------|------------------------------------|
| Python env setup            | uv, .venv                         | pip, conda, or platform default    |
| Install transformers        | ✅                                 | ✅                                 |
| Install Ollama              | ✅                                 | 🚫                                 |
| Install Unsloth/xformers    | 🚫                                 | ✅                                 |
| Data preparation            | ✅                                 | ✅                                 |
| Token extraction            | ✅                                 | ✅                                 |
| Fine-tuning/training        | 🚫                                 | ✅                                 |
| Quantized inference         | ✅ (Ollama)                        | ✅ (if needed)                     |
| GPU acceleration            | 🚫                                 | ✅                                 |

**Legend:** ✅ = Supported, 🚫 = Not supported

---

## 9. Troubleshooting Common Cloud Issues

| Problem | Solution |
|---------|----------|
| CUDA not found | Ensure you are using a GPU-enabled instance and drivers are installed |
| pip install fails | Upgrade pip: `pip install -U pip` |
| Out of memory (RAM/GPU) | Use smaller batch sizes, monitor usage, or upgrade instance |
| Unsloth/xformers import fails | Ensure you are in a cloud environment with CUDA, and install with pip |
| API key errors | Double-check `.env` or environment variables |

---

## 10. References
- [README: Environment Setup](../README.md#cloud-workflow-google-colab-lightning-etc)
- [Apple Silicon Setup Guide](apple_silicon_setup.md)
- [Token Extraction Guide](token_extraction.md)
- [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/en/index)

---

**For any issues not listed here, check the README or open an issue.** 
