"""
Data loader implementation for efficient loading and batching of template variations.

This module provides optimized data loading mechanisms that:
1. Minimize memory usage through lazy loading and streaming
2. Implement smart batching by grouping similar-length sequences
3. Support data splitting for train/val/test sets
4. Include performance monitoring and benchmarking
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Generator, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class BatchConfig:
    """Configuration for batch creation and processing."""
    batch_size: int = 32
    max_length: int = 512  # Maximum sequence length
    similar_length_tolerance: int = 50  # Max length difference for sequence grouping
    shuffle: bool = True

@dataclass
class DataStats:
    """Statistics about the loaded dataset."""
    total_examples: int
    avg_sequence_length: float
    length_distribution: Dict[int, int]
    template_distribution: Dict[str, int]

class TemplateDataLoader:
    """Efficient data loader for template variations with smart batching."""

    def __init__(
        self,
        data_dir: str | Path,
        batch_config: Optional[BatchConfig] = None,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)
    ):
        """
        Initialize the data loader.

        Args:
            data_dir: Directory containing the template variation data
            batch_config: Configuration for batch creation
            split_ratios: (train, val, test) ratios, must sum to 1.0
        """
        self.data_dir = Path(data_dir)
        self.batch_config = batch_config or BatchConfig()
        self.split_ratios = split_ratios
        self._validate_split_ratios()

        # Lazy loading - these will be populated when needed
        self._examples = None
        self._length_groups = None
        self._stats = None

        # Split indices - will be populated in _prepare_splits()
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

    def _validate_split_ratios(self):
        """Ensure split ratios are valid."""
        if not np.isclose(sum(self.split_ratios), 1.0):
            raise ValueError("Split ratios must sum to 1.0")
        if any(r < 0 or r > 1 for r in self.split_ratios):
            raise ValueError("Split ratios must be between 0 and 1")

    def _load_data(self) -> None:
        """Lazy load the data when first needed."""
        if self._examples is not None:
            return

        self._examples = []

        # Load template variations
        template_files = [
            f for f in self.data_dir.glob("*.json")
            if f.name in ["diverse_examples.json", "template_examples.json"]
        ]

        for file_path in template_files:
            with open(file_path) as f:
                data = json.load(f)
                # The examples are under the "examples" key
                if "examples" in data:
                    self._examples.extend(data["examples"])

        if not self._examples:
            raise ValueError(f"No examples found in {self.data_dir}")

        # Create length-based groups for efficient batching
        self._create_length_groups()

        # Prepare split indices
        self._prepare_splits()

        # Calculate statistics
        self._calculate_stats()

    def _create_length_groups(self) -> None:
        """Group examples by sequence length for efficient batching."""
        self._length_groups = defaultdict(list)

        for idx, example in enumerate(self._examples):
            # Get sequence length from the input field
            seq_length = len(example["input"])
            # Group into buckets based on tolerance
            bucket = seq_length // self.batch_config.similar_length_tolerance
            self._length_groups[bucket].append(idx)

    def _prepare_splits(self) -> None:
        """Prepare train/val/test split indices."""
        all_indices = list(range(len(self._examples)))
        if self.batch_config.shuffle:
            random.shuffle(all_indices)

        train_size = int(len(all_indices) * self.split_ratios[0])
        val_size = int(len(all_indices) * self.split_ratios[1])

        self.train_indices = all_indices[:train_size]
        self.val_indices = all_indices[train_size:train_size + val_size]
        self.test_indices = all_indices[train_size + val_size:]

    def _calculate_stats(self) -> None:
        """Calculate dataset statistics."""
        lengths = [len(ex["input"]) for ex in self._examples]

        template_counts = defaultdict(int)
        for ex in self._examples:
            template_counts[ex.get("template_category", "unknown")] += 1

        length_dist = defaultdict(int)
        for l in lengths:
            length_dist[l] += 1

        self._stats = DataStats(
            total_examples=len(self._examples),
            avg_sequence_length=np.mean(lengths),
            length_distribution=dict(length_dist),
            template_distribution=dict(template_counts)
        )

    def get_stats(self) -> DataStats:
        """Get statistics about the loaded dataset."""
        if self._stats is None:
            self._load_data()
        return self._stats

    def _get_batch_indices(self, indices: List[int]) -> Generator[List[int], None, None]:
        """Generate batches of indices with similar sequence lengths."""
        # Group indices by length bucket
        bucket_indices = defaultdict(list)
        for idx in indices:
            example = self._examples[idx]
            seq_length = len(example["input"])
            bucket = seq_length // self.batch_config.similar_length_tolerance
            bucket_indices[bucket].append(idx)

        # Create batches from each bucket
        for bucket in bucket_indices.values():
            if self.batch_config.shuffle:
                random.shuffle(bucket)

            for i in range(0, len(bucket), self.batch_config.batch_size):
                yield bucket[i:i + self.batch_config.batch_size]

    def get_batch(self, indices: List[int]) -> Dict:
        """Convert a batch of indices into a batch of examples."""
        batch_examples = [self._examples[i] for i in indices]

        # Create batch with proper padding
        max_length = min(
            max(len(ex["input"]) for ex in batch_examples),
            self.batch_config.max_length
        )

        # Create batch dictionary with all fields from the examples
        batch = {
            "inputs": [],
            "outputs": [],  # Changed from "targets" to match the JSON structure
            "template_categories": [],  # Changed from "template_types" to match the JSON structure
            "template_styles": [],
            "separator_styles": [],
            "metadata": []
        }

        for example in batch_examples:
            batch["inputs"].append(example["input"])
            batch["outputs"].append(example["output"])
            batch["template_categories"].append(example.get("template_category", "unknown"))
            batch["template_styles"].append(example.get("template_style"))
            batch["separator_styles"].append(example.get("separator_style", "none"))
            batch["metadata"].append({
                k: v for k, v in example.items()
                if k not in ["input", "output", "template_category", "template_style", "separator_style"]
            })

        return batch

    def train_batches(self) -> Generator[Dict, None, None]:
        """Generate batches for training."""
        self._load_data()
        for batch_indices in self._get_batch_indices(self.train_indices):
            yield self.get_batch(batch_indices)

    def val_batches(self) -> Generator[Dict, None, None]:
        """Generate batches for validation."""
        self._load_data()
        for batch_indices in self._get_batch_indices(self.val_indices):
            yield self.get_batch(batch_indices)

    def test_batches(self) -> Generator[Dict, None, None]:
        """Generate batches for testing."""
        self._load_data()
        for batch_indices in self._get_batch_indices(self.test_indices):
            yield self.get_batch(batch_indices)

    def get_example(self, idx: int) -> Dict:
        """Get a single example by index."""
        self._load_data()
        return self._examples[idx]
