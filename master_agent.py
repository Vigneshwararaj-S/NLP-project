import os
# Force all Hugging Face caches to your nobackup folder
os.environ["HF_HOME"] = "/nobackup/vigsa504/hf_cache"

import gc
import torch
from tqdm import tqdm
from PIL import Image
from transformers import (
    AutoProcessor, 
    AutoTokenizer, 
    Qwen2_5_VLForConditionalGeneration, 
    BitsAndBytesConfig
)
from peft import PeftModel

# ==========================================
# DEFAULT CLUSTER PATHS
# ==========================================
# ⚠️ CHANGE 'checkpoint-200' TO WHATEVER YOUR HIGHEST NUMBER IS:
LATEST_CHECKPOINT = "/nobackup/vigsa504/models/qwen-3b-inscription-lora-ft/checkpoint-300" 
OLD_TOKENIZER_PATH = "/nobackup/vigsa504/models/qwen-tamil-donor-untuned"
DEFAULT_NLP_MODEL_PATH = "/nobackup/vigsa504/models/qwen-tamil-segmenter-lora"

def run_vision_expert(image_paths, lora_path=LATEST_CHECKPOINT):
    """Pass 1: Loads 3B VLM, extracts raw text, and purges VRAM."""
    print(f"\n[Master Agent] Starting Vision OCR on {len(image_paths)} images using 3B Model...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 1. Load Pristine Processor
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=384 * 28 * 28
    )
    
    # 2. Vocab Stealing (to make sure the inference model can output Tamil!)
    tokenizer = processor.tokenizer
    old_tokenizer = AutoTokenizer.from_pretrained(OLD_TOKENIZER_PATH)
    old_vocab = set(old_tokenizer.get_vocab().keys())
    new_vocab = set(tokenizer.get_vocab().keys())
    custom_tokens = list(old_vocab - new_vocab)
    
    if custom_tokens:
        custom_tokens.sort()
        tokenizer.add_tokens(custom_tokens)
        
    processor.tokenizer = tokenizer
    
    # 3. Load 3B Base Model
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    # Expand the model's mouth to fit the custom tokens we just injected
    base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    # 4. Snap on your newly trained LoRA weights!
    print(f"Loading LoRA adapters from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    raw_texts = []
    for img_path in tqdm(image_paths, desc="OCR Inference"):
        try:
            image = Image.open(img_path).convert("RGB")
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract the raw continuous Tamil text from this stone inscription."}
                ]}
            ]
            
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            raw_texts.append(output_text)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            raw_texts.append("") 

    print("[Master Agent] Purging Vision Model from VRAM...")
    del model, base_model, processor
    gc.collect()
    torch.cuda.empty_cache()
    
    return raw_texts

# ==========================================
# LINGUISTIC EXPERT (Keep this exactly as you had it!)
# ==========================================
from transformers import AutoModelForCausalLM
def run_linguistic_expert(raw_texts, nlp_path=DEFAULT_NLP_MODEL_PATH):
    # ... Your existing segmenter code ...
    print(f"\n[Master Agent] Starting Linguistic Segmentation on {len(raw_texts)} strings...")
    
    tokenizer = AutoTokenizer.from_pretrained(nlp_path)
    model = AutoModelForCausalLM.from_pretrained(nlp_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    spaced_texts = []
    for raw_string in tqdm(raw_texts, desc="Sandhi Segmentation"):
        messages = [
            {"role": "system", "content": "You are a classical Tamil linguistic expert. Segment the following continuous string into correctly spaced words based on proper Sandhi rules."},
            {"role": "user", "content": raw_string}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        spaced_output = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
        spaced_texts.append(spaced_output)

    print("[Master Agent] Purging Linguistic Model from VRAM...")
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return spaced_texts

def process_inscriptions(image_paths):
    raw_texts = run_vision_expert(image_paths)
    final_texts = run_linguistic_expert(raw_texts)
    return final_texts