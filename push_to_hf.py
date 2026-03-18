import os
from huggingface_hub import HfApi

# ==========================================
# CONFIGURATION
# ==========================================
# ⚠️ Replace 'your_hf_username' with your actual Hugging Face username!
HF_USERNAME = "DonZdoN" 

# Local Paths on your cluster
VISION_LORA_PATH = "/nobackup/vigsa504/models/qwen-3b-inscription-lora-ft/checkpoint-300" # Update if you have a final merged folder
NLP_LORA_PATH = "/nobackup/vigsa504/models/qwen-tamil-segmenter-lora"
UNET_MODEL_PATH = "/nobackup/vigsa504/models/unet_stone_cleaner_fast.pth"

def main():
    api = HfApi()

    print("\n" + "="*50)
    print(" 🚀 UPLOADING PIPELINE TO HUGGING FACE")
    print("="*50)

    # ---------------------------------------------------------
    # 1. Upload Vision Expert (3B LoRA)
    # ---------------------------------------------------------
    vision_repo_id = f"{HF_USERNAME}/qwen2-vl-3b-vatteluttu-vision"
    print(f"\n📸 Creating repo: {vision_repo_id}...")
    api.create_repo(repo_id=vision_repo_id, exist_ok=True)
    
    print(f"Uploading Vision LoRA adapters from {VISION_LORA_PATH}...")
    api.upload_folder(
        folder_path=VISION_LORA_PATH,
        repo_id=vision_repo_id,
        commit_message="Upload Vatteluttu/Tamili Vision LoRA adapters"
    )
    print("✅ Vision Model Uploaded!")

    # ---------------------------------------------------------
    # 2. Upload Linguistic Expert (0.5B LoRA)
    # ---------------------------------------------------------
    nlp_repo_id = f"{HF_USERNAME}/qwen2.5-0.5b-tamil-segmenter"
    print(f"\n📝 Creating repo: {nlp_repo_id}...")
    api.create_repo(repo_id=nlp_repo_id, exist_ok=True)
    
    print(f"Uploading Segmenter LoRA adapters from {NLP_LORA_PATH}...")
    api.upload_folder(
        folder_path=NLP_LORA_PATH,
        repo_id=nlp_repo_id,
        commit_message="Upload Tamil Sandhi Segmentation LoRA adapters"
    )
    print("✅ Linguistic Model Uploaded!")

    # ---------------------------------------------------------
    # 3. Upload U-Net Denoising Model
    # ---------------------------------------------------------
    unet_repo_id = f"{HF_USERNAME}/unet-stone-denoiser"
    print(f"\n🧹 Creating repo: {unet_repo_id}...")
    api.create_repo(repo_id=unet_repo_id, exist_ok=True)
    
    print(f"Uploading U-Net weights from {UNET_MODEL_PATH}...")
    api.upload_file(
        path_or_fileobj=UNET_MODEL_PATH,
        path_in_repo="unet_stone_cleaner.pth",
        repo_id=unet_repo_id,
        commit_message="Upload U-Net stone denoising weights"
    )
    print("✅ U-Net Uploaded!")

    print("\n" + "="*50)
    print(" 🎉 ALL MODELS SUCCESSFULLY UPLOADED!")
    print(f" Check them out at: https://huggingface.co/{HF_USERNAME}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()