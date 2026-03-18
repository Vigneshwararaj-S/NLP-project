import os
import glob
from transformers import AutoTokenizer

def train_custom_tokenizer():
    # --- UPDATE THIS PATH TO YOUR CLEANED TEXT FOLDER ---
    data_dir = "complete_cleaned" 
    
    # Where the new tokenizer will be saved
    output_dir = "./qwen2.5-vl-tamil-tokenizer"
    base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    tokens_to_add = 5000

    print(f"1. Loading base tokenizer ({base_model_id})...")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"2. Locating text files in {data_dir}...")
    # Finds all .txt files even if they are inside sub-folders
    text_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    print(f"Found {len(text_files)} files to train on.")

    # Generator function to stream text to the trainer without crashing RAM
    def get_training_corpus():
        for filepath in text_files:
            with open(filepath, "r", encoding="utf-8") as f:
                # Yields chunks of text
                yield f.readlines()

    print(f"3. Training new BPE Tokenizer (Adding {tokens_to_add} Tamil tokens)...")
    print("This will take a minute or two as it calculates word frequencies.")
    
    # This magically merges Qwen's base vocabulary with your new Tamil tokens
    new_tokenizer = base_tokenizer.train_new_from_iterator(
        get_training_corpus(), 
        vocab_size=len(base_tokenizer) + tokens_to_add
    )

    print(f"4. Saving custom tokenizer to {output_dir}...")
    new_tokenizer.save_pretrained(output_dir)
    print("Success! Step 1 is complete. Your custom tokenizer is ready.")

if __name__ == "__main__":
    train_custom_tokenizer()