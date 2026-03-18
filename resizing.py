import os
# Force Hugging Face to use your 100GB drive for the massive 14GB download cache
os.environ["HF_HOME"] = "/nobackup/vigsa504/hf_cache"

import torch
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

def create_donor_model_ada():
    # Make sure this points to the tokenizer folder you created in Step 1
    tokenizer_path = "./qwen2.5-vl-tamil-tokenizer" 
    
    # Save the heavy, resized model securely in the new drive
    output_dir = "/nobackup/vigsa504/models/qwen-tamil-donor-untuned"
    
    print(f"1. Loading custom tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"2. Downloading {model_id} directly to Ada GPU via /nobackup cache...")
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto" 
    )

    print(f"3. Resizing model embeddings to match new tokenizer ({len(tokenizer)} tokens)...")
    model.resize_token_embeddings(len(tokenizer))

    print(f"4. Saving resized donor model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Success! The donor model is prepped and sitting safely on your 100GB drive.")

if __name__ == "__main__":
    create_donor_model_ada()