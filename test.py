import os
WORKSPACE_DIR = "/nobackup/vigsa504"
os.environ["HF_HOME"] = f"{WORKSPACE_DIR}/hf_cache"

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel
from PIL import Image

# --- PATHS ---
BASE_MODEL_PATH = f"{WORKSPACE_DIR}/models/qwen-tamil-donor-untuned"
# Point this to your saved output folder (or a specific checkpoint folder like checkpoint-100)
LORA_ADAPTER_PATH = f"{WORKSPACE_DIR}/models/qwen-inscription-lora-model-final" 

# ⚠️ CHANGE THIS TO A REAL IMAGE PATH YOU WANT TO TEST ⚠️
TEST_IMAGE_PATH = f"stone_Tholkapiyum_TSCII_vatteluttu_chunk_449.png" 

def run_inference():
    print("1. Loading Processor and Custom Tokenizer...")
    # FIX: Load from the base model instead of the empty LoRA folder!
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    # Inject your custom Tamil tokens just like we did in training
    added_tokens_path = f"{BASE_MODEL_PATH}/added_tokens.json"
    if os.path.exists(added_tokens_path):
        import json
        with open(added_tokens_path, "r") as f:
            added_tokens = json.load(f)
        sorted_tokens = [t[0] for t in sorted(added_tokens.items(), key=lambda x: x[1])]
        tokenizer.add_tokens(sorted_tokens)
        print(f"✅ Recovered and injected {len(sorted_tokens)} custom Tamil tokens.")
        
    tokenizer.pad_token_id = 151643
    tokenizer.pad_token = "<|endoftext|>"
    
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=512 * 28 * 28 
    )
    processor.tokenizer = tokenizer

    print("2. Loading Base Model...")
    # ... (Keep everything from Step 2 onwards exactly the same) ...

    print("2. Loading Base Model...")
    # Loading in bfloat16 for fast inference
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # CRITICAL: We must expand the base model's vocabulary matrix BEFORE attaching the adapter
    # Otherwise, the adapter won't fit onto the base model!
    base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    print("3. Attaching LoRA Adapter...")
    # This snaps your newly trained Vatteluttu weights onto the base model
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model.eval() # Put the model in "testing" mode

    print(f"4. Processing Image: {TEST_IMAGE_PATH}")
    image = Image.open(TEST_IMAGE_PATH).convert("RGB")
    
    # Format the prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": TEST_IMAGE_PATH},
                {"type": "text", "text": "Translate this ancient stone carving."} 
            ],
        }
    ]

    # The processor automatically inserts the exact correct number of <|image_pad|> tokens!
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Pass 'text' directly in, NO replacing!
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to("cuda")

    print("\n🚀 GENERATING TRANSLATION 🚀\n")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        
    # Trim the input prompt out of the generated IDs so it only prints the new answer
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    print("=========================================")
    print("MODEL OUTPUT:")
    print(output_text[0])
    print("=========================================")

if __name__ == "__main__":
    run_inference()