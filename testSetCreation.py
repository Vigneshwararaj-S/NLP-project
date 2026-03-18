import json
from datasets import load_dataset

# ==========================================
# PATHS
# ==========================================
# The file you just uploaded
FULL_DATASET_PATH = "segmentation_dataset_chunked.jsonl" 
# The new test file we are about to create
OUTPUT_TEST_PATH = "segmenter_test_dataset.jsonl"

def main():
    print(f"Loading full dataset from {FULL_DATASET_PATH}...")
    dataset = load_dataset("json", data_files=FULL_DATASET_PATH, split="train")
    
    print("Applying deterministic split (test_size=1000, seed=42)...")
    # This matches your training script exactly!
    split_dataset = dataset.train_test_split(test_size=1000, seed=42)
    
    test_set = split_dataset["test"]
    print(f"✅ Successfully isolated {len(test_set)} test samples.")
    
    print(f"Saving to {OUTPUT_TEST_PATH}...")
    with open(OUTPUT_TEST_PATH, "w", encoding="utf-8") as f:
        for item in test_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Done! Ready for evaluation.")

if __name__ == "__main__":
    main()