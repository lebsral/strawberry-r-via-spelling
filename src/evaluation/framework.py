"""
Core evaluation framework for assessing model performance on spelling-related tasks.

PROJECT POLICY: Qwen3-4B must ALWAYS be used in non-thinking mode (enable_thinking=False).
Any attempt to use thinking mode (enable_thinking=True) is prohibited and will raise an error.
See src/models/qwen3_loader.py for the enforced chat template utility.
"""

import os
import json
import yaml
import logging
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
import wandb

@dataclass
class EvaluationConfig:
    """Configuration for evaluation framework."""
    model_name: str
    max_new_tokens: int
    do_sample: bool
    num_beams: int
    batch_size: int
    max_examples: Optional[int]
    metrics: List[str]
    dataset_name: str
    dataset_split: str
    output_dir: str
    wandb_enabled: bool
    wandb_project: str
    wandb_run_name: str
    temperature: float = 0.6  # Qwen3-4B non-thinking default
    top_p: float = 0.95       # Qwen3-4B non-thinking default
    top_k: int = 20           # Qwen3-4B non-thinking default
    min_p: float = 0.0        # Qwen3-4B non-thinking default
    logging_level: str = "INFO"

    @classmethod
    def from_yaml(cls, config_path: str) -> 'EvaluationConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return cls(
            model_name=config['model']['name'],
            max_new_tokens=config['model']['max_new_tokens'],
            do_sample=config['model']['do_sample'],
            num_beams=config['model']['num_beams'],
            temperature=config['model'].get('temperature', 0.6),
            top_p=config['model'].get('top_p', 0.95),
            top_k=config['model'].get('top_k', 20),
            min_p=config['model'].get('min_p', 0.0),
            batch_size=config['evaluation']['batch_size'],
            max_examples=config['evaluation']['max_examples'],
            metrics=config['evaluation']['metrics'],
            dataset_name=config['data']['dataset_name'],
            dataset_split=config['data']['split'],
            output_dir=config['output']['base_dir'],
            wandb_enabled=config['logging']['wandb']['enabled'],
            wandb_project=config['logging']['wandb']['project'],
            wandb_run_name=config['logging']['wandb']['run_name'],
            logging_level=config['logging']['level']
        )

    def config_from_dict(self, config: Dict) -> 'EvaluationConfig':
        """Create EvaluationConfig from dictionary."""
        return EvaluationConfig(
            model_name=config['model']['name'],
            max_new_tokens=config['model']['max_new_tokens'],
            do_sample=config['model']['do_sample'],
            num_beams=config['model']['num_beams'],
            temperature=config['model'].get('temperature', 0.6),
            top_p=config['model'].get('top_p', 0.95),
            top_k=config['model'].get('top_k', 20),
            min_p=config['model'].get('min_p', 0.0),
            batch_size=config['evaluation']['batch_size'],
            max_examples=config['evaluation']['max_examples'],
            metrics=config['evaluation']['metrics'],
            dataset_name=config['data']['dataset_name'],
            dataset_split=config['data']['split'],
            output_dir=config['output']['base_dir'],
            wandb_enabled=config['logging']['wandb']['enabled'],
            wandb_project=config['logging']['wandb']['project'],
            wandb_run_name=config['logging']['wandb']['run_name'],
            logging_level=config['logging']['level']
        )

class BaseEvaluator(ABC):
    """Base class for evaluators.

    NOTE: For Qwen3-4B, only non-thinking mode is allowed for all inference and evaluation.
    If chat template formatting is used, always call apply_qwen3_chat_template_non_thinking from src/models/qwen3_loader.py.
    Thinking mode is strictly prohibited by project policy.
    """

    def __init__(self, config: EvaluationConfig, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not hasattr(self.model, 'device_map'):
            self.model = self.model.to(self.device)

        # Ensure model is in eval mode
        self.model.eval()

        # Enable memory efficient attention if available
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False

        # Set up logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, config.logging_level))

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info(f"Initialized evaluator with model: {config.model_name} on device: {self.device}")

    def generate_answer(self, question: str) -> str:
        """Generate model response for a question."""
        self.logger.debug(f"Generating answer for question: {question}")

        # Parameter validation
        assert 0.0 <= self.config.temperature <= 2.0, "temperature must be in [0.0, 2.0]"
        assert 0.0 <= self.config.top_p <= 1.0, "top_p must be in [0.0, 1.0]"
        assert self.config.top_k >= 0, "top_k must be >= 0"
        assert 0.0 <= self.config.min_p <= 1.0, "min_p must be in [0.0, 1.0]"

        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Tokenize with attention mask
            inputs = self.tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # Add max length to prevent excessive input
                return_attention_mask=True
            )

            # Move inputs to device if model doesn't have device_map
            if not hasattr(self.model, 'device_map'):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            self.logger.debug(f"Input shape: {inputs['input_ids'].shape}")

            with torch.inference_mode():  # Use inference mode for better memory handling
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    min_p=self.config.min_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            self.logger.debug(f"Output shape: {outputs.shape}")
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the generated part (remove the input)
            input_length = len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
            response = response[input_length:].strip()

            self.logger.debug(f"Generated response: {response}")

            # Clear memory
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return response

        except Exception as e:
            self.logger.error(f"Error during generation: {str(e)}")
            raise

    @abstractmethod
    def process_output(self, output: str) -> str:
        """Process model output into expected format."""
        pass

    @abstractmethod
    def evaluate(self, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        """Run evaluation on dataset."""
        pass

class EvaluationFramework:
    """Framework for running model evaluations."""

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """Initialize the framework with configuration."""
        self.evaluators = {}

        if config_path:
            self.config = EvaluationConfig.from_yaml(config_path)
        elif config_dict:
            self.config = self.config_from_dict(config_dict)
        else:
            raise ValueError("Either config_path or config_dict must be provided")

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(getattr(logging, self.config.logging_level))
        self.logger.info("Evaluation framework initialized")

    def config_from_dict(self, config: Dict) -> EvaluationConfig:
        """Create EvaluationConfig from dictionary."""
        return EvaluationConfig(
            model_name=config['model']['name'],
            max_new_tokens=config['model']['max_new_tokens'],
            do_sample=config['model']['do_sample'],
            num_beams=config['model']['num_beams'],
            temperature=config['model'].get('temperature', 0.6),
            top_p=config['model'].get('top_p', 0.95),
            top_k=config['model'].get('top_k', 20),
            min_p=config['model'].get('min_p', 0.0),
            batch_size=config['evaluation']['batch_size'],
            max_examples=config['evaluation']['max_examples'],
            metrics=config['evaluation']['metrics'],
            dataset_name=config['data']['dataset_name'],
            dataset_split=config['data']['split'],
            output_dir=config['output']['base_dir'],
            wandb_enabled=config['logging']['wandb']['enabled'],
            wandb_project=config['logging']['wandb']['project'],
            wandb_run_name=config['logging']['wandb']['run_name'],
            logging_level=config['logging']['level']
        )

    def register_evaluator(self, name: str, evaluator: BaseEvaluator) -> None:
        """Register an evaluator with the framework."""
        self.logger.info(f"Registering evaluator: {name}")
        self.evaluators[name] = evaluator

    def run_evaluation(self, data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Run evaluation using all registered evaluators."""
        results = {}
        for name, evaluator in self.evaluators.items():
            self.logger.info(f"Running evaluation with {name}")
            results[name] = evaluator.evaluate(data)
        return results

if __name__ == "__main__":
    config = EvaluationConfig.from_yaml("configs/evaluation/base_config.yaml")
    print("Loaded config:")
    print(config)
    # Optionally, instantiate a dummy model/tokenizer and call generate_answer
