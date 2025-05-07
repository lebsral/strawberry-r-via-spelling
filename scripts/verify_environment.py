#!/usr/bin/env python3
"""
Environment verification script to ensure all components are properly installed and configured.
"""

import sys
import os
from pathlib import Path
import importlib
import subprocess
import json

def check_python_version():
    print(f"Python version: {sys.version}")
    assert sys.version_info >= (3, 10), "Python version must be 3.10 or higher"
    print("✅ Python version check passed")

def check_required_packages():
    required_packages = [
        'torch',
        'transformers',
        'datasets',
        'wandb',
        'dspy',
        'lightning',
        'matplotlib',
        'seaborn',
        'pandas',
        'jupyter',
        'notebook',
        'ipywidgets'
    ]

    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError as e:
            print(f"❌ Failed to import {package}: {e}")
            raise

def check_env_file():
    env_example = Path('.env.example')
    env_file = Path('.env')

    assert env_example.exists(), ".env.example file is missing"
    assert env_file.exists(), ".env file is missing"
    print("✅ Environment files check passed")

def check_directory_structure():
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/splits',
        'notebooks',
        'src/data',
        'src/models',
        'src/training',
        'src/evaluation',
        'scripts',
        'configs',
        'tests',
        'results',
        'checkpoints',
        'tasks'
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        assert path.exists(), f"Directory {dir_path} is missing"
        assert path.is_dir(), f"{dir_path} is not a directory"
    print("✅ Directory structure check passed")

def check_git_setup():
    try:
        subprocess.run(['git', 'status'], check=True, capture_output=True)
        print("✅ Git repository is properly initialized")
    except subprocess.CalledProcessError:
        print("❌ Git repository is not properly initialized")
        raise

def check_requirements_file():
    req_file = Path('requirements.txt')
    assert req_file.exists(), "requirements.txt file is missing"
    print("✅ requirements.txt exists")

def check_readme():
    readme = Path('README.md')
    assert readme.exists(), "README.md file is missing"
    print("✅ README.md exists")

def main():
    print("Starting environment verification...")
    print("-" * 50)

    try:
        check_python_version()
        check_required_packages()
        check_env_file()
        check_directory_structure()
        check_git_setup()
        check_requirements_file()
        check_readme()

        print("-" * 50)
        print("✅ All verification checks passed!")
        return 0

    except AssertionError as e:
        print(f"\n❌ Verification failed: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error during verification: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
