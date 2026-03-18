import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer

# --- PATHS ---
WORKSPACE_DIR = "/nobackup/vigsa504"
os.environ["HF_HOME"] = f"{WORKSPACE_DIR}/hf_cache"

DATASET_PATH = f"{WORKSPACE_DIR}/data/segmentation_dataset_chunked.jsonl"
OUTPUT_DIR = f"{WORKSPACE_DIR}/models/qwen-tamil-segmenter-lora"

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

def setup_and_train():
    print("1. Loading Tokenizer and Lightweight Model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Loading natively in 16-bit (No QLoRA needed for a 0.5B model!)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print("2. Configuring LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("3. Loading and Splitting Chunked Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # Reserving 1000 examples for validation out of the 25,611 chunks
    split_dataset = dataset.train_test_split(test_size=1000, seed=42)

    print("4. Initializing Text-Only SFT Trainer...")
    from trl import SFTConfig

    # Define all training arguments directly inside SFTConfig
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,        
        gradient_accumulation_steps=4,        
        learning_rate=3e-4,                   
        num_train_epochs=2,
        logging_steps=50,                     
        save_strategy="epoch",  
        eval_strategy="epoch",          
        bf16=True,                            
        optim="adamw_torch",
        gradient_checkpointing=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=split_dataset["train"],      
        eval_dataset=split_dataset["test"],         
        processing_class=tokenizer
    )

    print("\n🚀 STARTING TEXT SEGMENTATION TRAINING 🚀")
    trainer.train()

    print(f"\n✅ Training Complete! Saving weights to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    setup_and_train()