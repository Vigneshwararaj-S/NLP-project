import os
import sys

# Force all caches to the nobackup drive
os.environ["HF_HOME"] = "/nobackup/vigsa504/hf_cache"
os.environ["TORCH_HOME"] = "/nobackup/vigsa504/torch_cache"

import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp

# Import your two experts from the master agent
from master_agent import run_vision_expert, run_linguistic_expert

# ==========================================
# CONFIGURATION
# ==========================================
# ⚠️ Change this to any specific stone image chunk you want to test!
TEST_IMAGE_PATH = "dataset-card.jpg" # Update this to your test image

UNET_MODEL_PATH = "/nobackup/vigsa504/models/unet_stone_cleaner_fast.pth" 
CLEANED_IMAGE_OUTPUT = "cleaned_test_mask.png"

def clean_image_with_unet(image_path, model_path, output_path):
    print(f"\n🧹 STAGE 1: U-Net Denoising -> {image_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load U-Net Model (ResNet34 backbone)
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights=None, 
        in_channels=3, 
        classes=1, 
        activation="sigmoid"
    )
    
    if not os.path.exists(model_path):
        print(f"❌ Error: U-Net model not found at {model_path}")
        sys.exit(1)
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load and Preprocess Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Could not read image at {image_path}")
        sys.exit(1)
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = (img.shape[1], img.shape[0]) # (Width, Height)

    # Resize to the standard 1024x256 used during training
    img_resized = cv2.resize(img, (1024, 256))
    img_normalized = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    # 3. Predict Mask
    with torch.no_grad():
        pred = model(tensor)

    # 4. Post-process to purely black/white pixels
    mask = pred.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255 

    # Resize back to the original stone image's dimensions so the Vision model doesn't get distorted shapes
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    # Save to disk
    cv2.imwrite(output_path, mask_resized)
    print(f"✅ Cleaned mask saved to {output_path}")
    
    return output_path

def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: Test image not found at {TEST_IMAGE_PATH}")
        sys.exit(1)

    print("\n" + "="*60)
    print(" 🚀 STARTING FULL END-TO-END OCR PIPELINE")
    print("="*60)

    # --- STAGE 1: U-NET ---
    clean_img_path = clean_image_with_unet(TEST_IMAGE_PATH, UNET_MODEL_PATH, CLEANED_IMAGE_OUTPUT)

    # --- STAGE 2: VISION EXPERT (3B) ---
    print("\n📸 STAGE 2: Analyzing Clean Mask with Vision Model...")
    vision_results = run_vision_expert([clean_img_path])
    raw_continuous_text = vision_results[0]
    
    print("\n" + "-"*40)
    print("🧠 VISION MODEL OUTPUT (Raw & Continuous):")
    print("-" * 40)
    print(raw_continuous_text)

    # --- STAGE 3: LINGUISTIC EXPERT (0.5B) ---
    print("\n📝 STAGE 3: Applying Sandhi Segmentation...")
    linguistic_results = run_linguistic_expert([raw_continuous_text])
    final_spaced_text = linguistic_results[0]

    print("\n" + "="*60)
    print(" ✨ FINAL PIPELINE OUTPUT (Modern Spacing) ✨")
    print("="*60)
    print(final_spaced_text)
    print("="*60 + "\n")

if __name__ == "__main__":
    main()