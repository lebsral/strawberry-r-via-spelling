"""Test script for validating the template data loader functionality."""

import sys
from pathlib import Path
import time

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.data_loader import TemplateDataLoader, BatchConfig

def test_data_loader():
    """Run tests to validate data loader functionality."""

    # Initialize data loader
    data_dir = project_root / "data" / "processed" / "template_variations"
    batch_config = BatchConfig(
        batch_size=16,
        max_length=256,
        similar_length_tolerance=25,
        shuffle=True
    )

    loader = TemplateDataLoader(
        data_dir=data_dir,
        batch_config=batch_config,
        split_ratios=(0.8, 0.1, 0.1)
    )

    # Test 1: Data Loading and Statistics
    print("\n=== Testing Data Loading and Statistics ===")
    stats = loader.get_stats()
    print(f"Total examples: {stats.total_examples}")
    print(f"Average sequence length: {stats.avg_sequence_length:.2f}")
    print("\nTemplate distribution:")
    for template, count in stats.template_distribution.items():
        print(f"  {template}: {count}")

    # Test 2: Batch Generation Performance
    print("\n=== Testing Batch Generation Performance ===")
    start_time = time.time()
    batch_count = 0
    total_examples = 0

    for batch in loader.train_batches():
        batch_count += 1
        total_examples += len(batch["inputs"])

        # Print progress for first few batches
        if batch_count <= 3:
            print(f"\nBatch {batch_count}:")
            print(f"  Number of examples: {len(batch['inputs'])}")
            print(f"  Template categories: {set(batch['template_categories'])}")
            print(f"  Separator styles: {set(batch['separator_styles'])}")

        if batch_count == 1:
            # Detailed inspection of first batch
            print("\nDetailed first batch inspection:")
            print(f"  Keys in batch: {list(batch.keys())}")
            print(f"  First example input: {batch['inputs'][0][:100]}...")
            print(f"  First example output: {batch['outputs'][0][:100]}...")
            print(f"  First example template category: {batch['template_categories'][0]}")
            print(f"  First example separator style: {batch['separator_styles'][0]}")

    duration = time.time() - start_time
    print(f"\nProcessing Summary:")
    print(f"  Total batches: {batch_count}")
    print(f"  Total examples: {total_examples}")
    print(f"  Processing time: {duration:.2f} seconds")
    print(f"  Examples per second: {total_examples/duration:.2f}")

    # Test 3: Data Split Verification
    print("\n=== Testing Data Split Ratios ===")
    train_count = sum(1 for _ in loader.train_batches())
    val_count = sum(1 for _ in loader.val_batches())
    test_count = sum(1 for _ in loader.test_batches())

    print("Batch counts per split:")
    print(f"  Train: {train_count}")
    print(f"  Validation: {val_count}")
    print(f"  Test: {test_count}")

    # Test 4: Memory Usage
    print("\n=== Testing Memory Efficiency ===")
    import psutil
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB

    # Force some data loading
    for _ in loader.train_batches():
        pass

    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Memory usage:")
    print(f"  Before loading: {memory_before:.2f} MB")
    print(f"  After loading: {memory_after:.2f} MB")
    print(f"  Difference: {memory_after - memory_before:.2f} MB")

if __name__ == "__main__":
    test_data_loader()
