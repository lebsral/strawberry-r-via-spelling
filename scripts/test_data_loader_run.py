from src.data.data_loader import TemplateDataLoader, BatchConfig
from pathlib import Path

if __name__ == "__main__":
    loader = TemplateDataLoader(
        data_dir=Path("data/processed"),
        batch_config=BatchConfig(batch_size=2, max_length=128, shuffle=False)
    )
    stats = loader.get_stats()
    print("Stats:", stats)
    for batch in loader.train_batches():
        print("Batch:", batch)
        break  # Just show the first batch
