from src.evaluation.utils import load_yaml, save_json, load_json, batch_generator

if __name__ == "__main__":
    # Test JSON save/load
    data = {"foo": 1, "bar": [1, 2, 3]}
    save_json(data, "results/test_utils.json")
    loaded = load_json("results/test_utils.json")
    print("Loaded JSON:", loaded)

    # Test batch generator
    items = list(range(10))
    for batch in batch_generator(items, batch_size=3):
        print("Batch:", batch)
