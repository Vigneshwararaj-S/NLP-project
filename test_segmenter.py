import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# --- SETTINGS ---
WORKSPACE_DIR = "/nobackup/vigsa504"
BASE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
# Point this to your saved Stage 2 weights
LORA_PATH = f"{WORKSPACE_DIR}/models/qwen-tamil-segmenter-lora"

def load_segmenter():
    print("1. Loading Base Model and Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
    
    # Load base model in bfloat16 for speed
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("2. Attaching Tamil Segmentation Adapters...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()
    return model, tokenizer

def segment_text(model, tokenizer, unspaced_text):
    # Construct the exact same prompt format used in training
    messages = [
        {"role": "system", "content": "You are a Tamil linguistic expert. Insert the correct modern spaces into this continuous ancient text."},
        {"role": "user", "content": unspaced_text}
    ]
    
    # Apply the chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("\n📝 SEGMENTING TEXT... 📝\n")
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,  # Keep it deterministic for the best logical splits
            repetition_penalty=1.1
        )
    
    # Decode only the newly generated tokens
    response = tokenizer.decode(output_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    model, tokenizer = load_segmenter()
    
    # TEST DATA: Try a string the model hasn't seen, or one of your unspaced strings
    test_string = "வெண்குடைஒக்குமால்எனவேமலைந்தோரும்பணிந்தோரும்பாடியவர்கோனாட்டுஎறிச்சிலுர்மாடலன்மதுரைக்குமரனார்பாடப்பட்டோன்சோழன்இலவந்திகைப்பள்ளித்" 
    
    result = segment_text(model, tokenizer, test_string)
    
    print(f"Original: {test_string}")
    print(f"Segmented: {result}")
    print("-" * 50)

    