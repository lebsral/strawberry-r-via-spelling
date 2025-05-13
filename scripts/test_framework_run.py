from src.evaluation.framework import EvaluationFramework, BaseEvaluator, EvaluationConfig
from typing import List, Dict, Any

class DummyEvaluator(BaseEvaluator):
    def __init__(self, config, model, tokenizer):
        # Do not call super().__init__ to avoid model.to()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.logger = None
    def process_output(self, output: str) -> str:
        return output
    def evaluate(self, dataset: List[Dict[str, str]]) -> Dict[str, Any]:
        return {"dummy_metric": {"value": 1.0}}

if __name__ == "__main__":
    # Minimal config dict (adjust as needed)
    config_dict = {
        "model": {"name": "dummy", "max_new_tokens": 10, "do_sample": False, "num_beams": 1},
        "evaluation": {"batch_size": 2, "max_examples": 2, "metrics": []},
        "data": {"dataset_name": "dummy", "split": "test"},
        "output": {"base_dir": "results/"},
        "logging": {"wandb": {"enabled": False, "project": "dummy", "run_name": "dummy"}, "level": "INFO"}
    }
    framework = EvaluationFramework(config_dict=config_dict)
    framework.register_evaluator("dummy", DummyEvaluator(framework.config, None, None))
    # Minimal valid Alpaca-format data
    data = [
        {"instruction": "Spell 'apple'", "input": "apple", "output": "a p p l e"},
        {"instruction": "Spell 'banana'", "input": "banana", "output": "b a n a n a"}
    ]
    results = framework.run_evaluation(data)
    print("Results:", results)
