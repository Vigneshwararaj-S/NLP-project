import os

# --- CRITICAL CACHE & MEMORY OVERRIDES ---
WORKSPACE_DIR = "/nobackup/vigsa504"
os.environ["HF_HOME"] = f"{WORKSPACE_DIR}/hf_cache"
os.environ["HF_DATASETS_CACHE"] = f"{WORKSPACE_DIR}/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{WORKSPACE_DIR}/hf_cache/models"
os.environ["TMPDIR"] = f"{WORKSPACE_DIR}/pip_tmp"
# THIS PREVENTS VRAM FRAGMENTATION CRASHES:
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments
from PIL import Image
from transformers import BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

MODEL_PATH = f"{WORKSPACE_DIR}/models/qwen-tamil-donor-untuned"
DATASET_PATH = f"{WORKSPACE_DIR}/data/fixed_dataset.jsonl"
OUTPUT_DIR = f"{WORKSPACE_DIR}/models/qwen-inscription-lora-model-final"

def process_images_and_labels(example):
    script_label = "unknown"
    image_paths = []
    
    for message in example["messages"]:
        for content in message["content"]:
            if content["type"] == "image":
                img_path = content["image"]
                if "tamili" in img_path.lower():
                    script_label = "tamili"
                elif "vatteluttu" in img_path.lower():
                    script_label = "vatteluttu"
                
                # LAZY LOADING
                image_paths.append(img_path)
    
    example["script_type"] = script_label
    example["image_paths"] = image_paths 
    return example

def setup_and_train():
    print("1. Loading Pristine Processor and Tokenizer...")
    # ENFORCING MAX PIXELS: Dropped to 448 to save activation memory on the GPU!
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=256 * 28 * 28,
        max_pixels=448 * 28 * 28 
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    
    added_tokens_path = f"{MODEL_PATH}/added_tokens.json"
    if os.path.exists(added_tokens_path):
        import json
        with open(added_tokens_path, "r") as f:
            added_tokens = json.load(f)
        sorted_tokens = [t[0] for t in sorted(added_tokens.items(), key=lambda x: x[1])]
        tokenizer.add_tokens(sorted_tokens)
        print(f"✅ Recovered and injected {len(sorted_tokens)} custom Tamil tokens.")
        
    tokenizer.pad_token_id = 151643
    tokenizer.pad_token = "<|endoftext|>"
    processor.tokenizer = tokenizer
    
    print("2. Loading the Model (4-bit QLoRA Optimized)...")
    # --- THE SILVER BULLET: 4-BIT QUANTIZATION ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=bnb_config  # Loads the model in 5GB instead of 15GB!
    )
    
    # Sync model config with repaired tokenizer
    model.config.pad_token_id = 151643
    model.config.eos_token_id = 151645
    
    # Expand the GPU's embedding matrix
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    print("3. Configuring LoRA and Unfreezing Embeddings...")
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # DROPPED LM_HEAD: This cuts trainable params from 1.1B to 550M, saving ~2.5GB of VRAM!
        modules_to_save=["embed_tokens"] 
    )

    # --- ADD THESE TWO MISSING LINES BACK ---
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # ----------------------------------------

    print("4. Loading, Mapping, and Stratifying the Dataset...")

    print("4. Loading, Mapping, and Stratifying the Dataset...")
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    dataset = dataset.map(
        process_images_and_labels,
        num_proc=16,              
        writer_batch_size=100,    
        desc="Loading images into HF Cache"
    )

    dataset = dataset.class_encode_column("script_type")
    split_dataset = dataset.train_test_split(
        train_size=5000, 
        test_size=1000, 
        stratify_by_column="script_type", 
        seed=42
    )
    
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]
    print(f"✅ Training on {len(train_data)} images. Validating on {len(val_data)} images.")

    def custom_collate_fn(features):
        texts = []
        for f in features:
            clean_messages = []
            for msg in f["messages"]:
                if isinstance(msg["content"], list):
                    clean_content = []
                    for content in msg["content"]:
                        if content["type"] == "text":
                            clean_txt = content["text"].replace("<image>", "").replace("<|image_pad|>", "").strip()
                            clean_content.append({"type": "text", "text": clean_txt})
                        else:
                            clean_content.append(content)
                    clean_messages.append({"role": msg["role"], "content": clean_content})
                else:
                    clean_txt = str(msg["content"]).replace("<image>", "").replace("<|image_pad|>", "").strip()
                    clean_messages.append({"role": msg["role"], "content": clean_txt})
                    
            texts.append(processor.apply_chat_template(clean_messages, tokenize=False))
            
        images = []
        for f in features:
            if "image_paths" in f and f["image_paths"]:
                images.extend([Image.open(p).convert("RGB") for p in f["image_paths"]])
                
        batch = processor(
            text=texts,
            images=images if len(images) > 0 else None,
            padding=True,
            return_tensors="pt"
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        
        return batch

    print("5. Initializing Trainer...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,        
        gradient_accumulation_steps=16,       
        learning_rate=2e-4,                   
        num_train_epochs=3,                   # Kept at 3 epochs per your request!
        logging_steps=5,                      
        
        # --- ALIGNED CHECKPOINTING FOR VPN DROPS ---
        save_strategy="steps",                # Safely saves progress during long 3-epoch runs
        save_steps=100,               
        save_total_limit=3,                        
        eval_strategy="steps",                # Aligned evaluation to match the saving steps
        eval_steps=100,
        per_device_eval_batch_size=1,   
        bf16=True,    
        
        # --- SPEED BOOST ---
        gradient_checkpointing=True,          
        optim="paged_adamw_8bit",                   # Removed 'paged_' so the GPU runs at maximum speed
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_pin_memory=False,          
        
        remove_unused_columns=False           
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,      
        eval_dataset=val_data,         
        data_collator=custom_collate_fn, 
    )

    print("\n🚀 STARTING TRAINING LOOP 🚀")
    # --- SMART AUTO-RESUME ---
    from transformers.trainer_utils import get_last_checkpoint
    
    # Check if a save file already exists in your output folder
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)

    if last_checkpoint is not None:
        print(f"\n🔄 VPN DROP DETECTED. RESUMING EXACTLY FROM: {last_checkpoint} 🔄")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\n🚀 STARTING FRESH TRAINING LOOP 🚀")
        trainer.train()

    print(f"\n✅ Training Complete! Saving LoRA weights to: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)


    processor.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    torch.cuda.empty_cache()
    setup_and_train()


############################### --- LEGACY CODE (BEFORE FIXES) --- ###############################

# import os


# WORKSPACE_DIR = "/nobackup/vigsa504"
# os.environ["HF_HOME"] = f"{WORKSPACE_DIR}/hf_cache"
# os.environ["HF_DATASETS_CACHE"] = f"{WORKSPACE_DIR}/hf_cache/datasets"
# os.environ["TRANSFORMERS_CACHE"] = f"{WORKSPACE_DIR}/hf_cache/models"
# os.environ["TMPDIR"] = f"{WORKSPACE_DIR}/pip_tmp"

# import torch
# from datasets import load_dataset, ClassLabel
# from peft import LoraConfig, get_peft_model
# from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
# from trl import SFTTrainer, SFTConfig
# from PIL import Image


# MODEL_PATH = f"{WORKSPACE_DIR}/models/qwen-tamil-donor-untuned"
# DATASET_PATH = f"{WORKSPACE_DIR}/data/fixed_dataset.jsonl"
# OUTPUT_DIR = f"{WORKSPACE_DIR}/models/qwen-vatteluttu-lora-final"


# def process_images_and_labels(example):
#     """
#     1. Loads the PIL images to prevent TRL from crashing.
#     2. Scans the filename to tag the row as 'tamili' or 'vatteluttu' for stratified splitting.
#     """
#     images = []
#     script_label = "unknown"
    
#     for message in example["messages"]:
#         for content in message["content"]:
#             if content["type"] == "image":
#                 img_path = content["image"]
                
#                 # Check the filename for your specific naming convention
#                 if "tamili" in img_path.lower():
#                     script_label = "tamili"
#                 elif "vatteluttu" in img_path.lower():
#                     script_label = "vatteluttu"
                
#                 # Convert to RGB to ensure 3-channel format
#                 images.append(Image.open(img_path).convert("RGB"))
    
#     example["images"] = images
#     example["script_type"] = script_label
#     return example

# def setup_and_train():
#     print("1. Loading Processor and Custom Tokenizer...")
#     processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
#     processor.tokenizer = tokenizer
    
#     print("2. Loading the Model (Ada Optimized)...")
#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch.bfloat16,
#         device_map="auto"
#     )

#     print("3. Configuring LoRA and Unfreezing Embeddings...")
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         lora_dropout=0.05,
#         bias="none",
#         task_type="CAUSAL_LM",
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         modules_to_save=["embed_tokens", "lm_head"] 
#     )

#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()

#     print("4. Loading, Mapping, and Stratifying the Dataset...")
#     dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
#     # Apply the mapping function using 16 CPU cores to bypass the bottleneck
#     dataset = dataset.map(
#         process_images_and_labels,
#         num_proc=16,              
#         writer_batch_size=100,    
#         desc="Loading images into HF Cache"
#     )

#     # Hugging Face requires labels to be 'ClassLabel' types to stratify them
#     dataset = dataset.class_encode_column("script_type")

#     # Perform the Stratified Split. Since your input is perfectly balanced, the output is perfectly balanced!
#     split_dataset = dataset.train_test_split(
#         test_size=0.2, 
#         stratify_by_column="script_type", 
#         seed=42
#     )
    
#     train_data = split_dataset["train"]
#     val_data = split_dataset["test"]

#     print(f"✅ Training on {len(train_data)} images. Validating on {len(val_data)} images.")

#     print("5. Initializing SFT Trainer...")
#     training_args = SFTConfig(
#         output_dir=OUTPUT_DIR,
#         per_device_train_batch_size=1,        
#         gradient_accumulation_steps=8,        
#         learning_rate=2e-4,                   
#         num_train_epochs=3,                   
#         logging_steps=10,                     
#         save_strategy="epoch",  
        
#         eval_strategy="epoch",          
#         per_device_eval_batch_size=1,   
        
#         bf16=True,                            
#         gradient_checkpointing=True,          
#         optim="adamw_torch_fused",            
#         dataset_text_field="messages",        
#         remove_unused_columns=False           
#     )

#     trainer = SFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_data,      
#         eval_dataset=val_data,         
#         processing_class=processor,
#     )

#     print("\n🚀 STARTING TRAINING LOOP 🚀")
#     trainer.train()

#     print(f"\n✅ Training Complete! Saving LoRA weights to: {OUTPUT_DIR}")
#     trainer.save_model(OUTPUT_DIR)

# if __name__ == "__main__":
#     torch.cuda.empty_cache()
#     setup_and_train()


