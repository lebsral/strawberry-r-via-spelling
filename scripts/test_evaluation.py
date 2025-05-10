"""
Test script to verify the evaluation framework.
"""

import os
import sys
import logging
import gc
from pathlib import Path
import torch

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.framework import EvaluationFramework, BaseEvaluator
from src.evaluation.metrics import metrics_registry
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestEvaluator(BaseEvaluator):
    """Simple evaluator for testing the framework."""

    def __init__(self, config, model, tokenizer):
        super().__init__(config, model, tokenizer)
        # Ultra-simple prompt focusing on exact pattern matching
        self.example_prompt = """Count letters or find position. Answer with single number or letter.

Question: How many 'a's in 'cat'?
Answer: 1

Question: {question}
Answer: """

    def process_output(self, output: str) -> str:
        """Process model output to extract relevant information."""
        # Extract just the first character/number
        output = output.strip().lower()
        if not output:
            return "0"
        return output[0]

    def generate_answer(self, question: str) -> str:
        """Generate answer with examples."""
        try:
            # Remove the "Answer with just the letter:" part if present
            question = question.replace(" Answer with just the letter:", "")
            prompt = self.example_prompt.format(question=question)

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,  # Just one token
                    num_beams=1,
                    do_sample=False,  # Pure greedy
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the generated part after our prompt
            return response[len(prompt):]
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "ERROR"

    def evaluate_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single example."""
        try:
            question = example["question"]
            expected = example["answer"]

            # Generate and process the model's answer
            model_output = self.generate_answer(question)
            processed_output = self.process_output(model_output)

            # Calculate metrics
            metrics = {
                "exact_match": processed_output.lower() == expected.lower(),
                "model_output": processed_output,
                "expected_output": expected
            }

            return metrics
        except Exception as e:
            logger.error(f"Error evaluating example: {str(e)}")
            return {
                "error": str(e),
                "exact_match": False,
                "model_output": "ERROR",
                "expected_output": example["answer"]
            }

    def evaluate(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run evaluation on the entire dataset."""
        results = []
        for example in dataset:
            result = self.evaluate_example(example)
            results.append(result)

        # Calculate overall metrics
        total_examples = len(results)
        correct_answers = sum(1 for r in results if r["exact_match"])
        accuracy = correct_answers / total_examples if total_examples > 0 else 0

        return {
            "accuracy": accuracy,
            "total_examples": total_examples,
            "correct_answers": correct_answers,
            "detailed_results": results
        }

def main():
    try:
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create a simple configuration
        config = {
            "model": {
                "name": "Qwen/Qwen3-4B",
                "max_new_tokens": 5,
                "temperature": 0.1,
                "do_sample": True,
                "num_beams": 1
            },
            "evaluation": {
                "batch_size": 1,
                "max_examples": None,
                "metrics": ["exact_match"]
            },
            "data": {
                "dataset_name": "test",
                "split": "test"
            },
            "output": {
                "base_dir": "results/evaluation"
            },
            "logging": {
                "level": "INFO",
                "wandb": {
                    "enabled": False,
                    "project": "test",
                    "run_name": "test"
                }
            }
        }

        logger.info("Loading model and tokenizer...")
        model_name = config["model"]["name"]
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Ensure the model has padding and EOS tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # Create some test data
        test_data = [
            {
                "question": "How many 'a's are there in the word 'evaluation'?",
                "answer": "2"
            },
            {
                "question": "What is the first letter in 'world'? Answer with just the letter:",
                "answer": "w"
            },
            {
                "question": "How many 't's are there in the word 'python'?",
                "answer": "1"
            },
            {
                "question": "What is the third letter in 'coding'? Answer with just the letter:",
                "answer": "d"
            },
            {
                "question": "How many 'l's are there in the word 'hello'?",
                "answer": "2"
            }
        ]

        logger.info("Initializing evaluation framework...")
        framework = EvaluationFramework(config_dict=config)

        # Register the test evaluator
        evaluator = TestEvaluator(framework.config, model, tokenizer)
        framework.register_evaluator("test_evaluator", evaluator)

        logger.info("\nRunning evaluation...")
        results = []
        for example in test_data:
            try:
                result = evaluator.evaluate_example(example)
                results.append(result)
                logger.info(f"\nQuestion: {example['question']}")
                logger.info(f"Expected: {result['expected_output']}")
                logger.info(f"Got: {result['model_output']}")
                logger.info(f"Exact Match: {result['exact_match']}")
            except Exception as e:
                logger.error(f"Error processing example: {str(e)}")
                continue

            # Force garbage collection between examples
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate overall accuracy
        accuracy = sum(r["exact_match"] for r in results) / len(results)
        logger.info(f"\nOverall Accuracy: {accuracy:.2%}")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
