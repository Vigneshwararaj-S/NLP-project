import os
import json
import re
from datasets import load_dataset

# ==========================================
# PATHS
# ==========================================
# Ensure this is the dataset file that contains the CLEAN image names
DATASET_PATH = "/nobackup/vigsa504/data/fixed_dataset.jsonl" 
OUTPUT_PATH = "/home/vigsa504/NLP project/test_dataset.jsonl"

# WHERE YOUR CLEAN IMAGES ACTUALLY LIVE ON THE CLUSTER
VATTELUTTU_DIR = "/nobackup/vigsa504/data/chunked_images/vatteluttu/"
TAMILI_DIR = "/nobackup/vigsa504/data/chunked_images/tamili/"

def process_metadata_and_paths(example):
    """Updates paths to cluster Linux format and extracts script type."""
    messages = example["messages"]
    image_path = ""
    
    # 1. Locate the image entry in the messages
    for item in messages[0]["content"]:
        if item.get("type") == "image":
            old_path = item["image"]
            # Extract just the filename (e.g., Iniyavai40_tamili_chunk_1.png)
            filename = re.split(r'[\\/]', old_path)[-1]
            
            # 2. Route to the correct Linux directory
            if "vatteluttu" in filename.lower():
                script_type = "vatteluttu"
                new_path = os.path.join(VATTELUTTU_DIR, filename)
            elif "tamili" in filename.lower():
                script_type = "tamili"
                new_path = os.path.join(TAMILI_DIR, filename)
            else:
                script_type = "unknown"
                new_path = old_path
            
            # Update the path inside the message content
            item["image"] = new_path
            image_path = new_path
            break
            
    return {"script_type": script_type, "image_path": image_path}

def main():
    print("Loading full dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    print("Updating image paths to cluster format and tagging script types...")
    dataset = dataset.map(process_metadata_and_paths)
    
    print("Encoding columns for stratification...")
    dataset = dataset.class_encode_column("script_type")

    print("Executing Stratified Split (Seed 42)...")
    # This MUST match the split parameters used in your training.py
    split_dataset = dataset.train_test_split(
        train_size=5000, 
        test_size=1000, 
        stratify_by_column="script_type", 
        seed=42
    )
    
    test_set = split_dataset["test"]
    print(f"✅ Isolated {len(test_set)} test samples with corrected cluster paths.")
    
    # Optional: Verify the first file actually exists to prevent [Errno 2] later
    sample_path = test_set[0]["image_path"]
    if os.path.exists(sample_path):
        print(f"✅ Path verification successful: Found {sample_path}")
    else:
        print(f"❌ Path verification failed: {sample_path} not found. Check your directory variables!")

    print(f"Saving to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in test_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print("Done! Ready for evaluation.")

if __name__ == "__main__":
    main()