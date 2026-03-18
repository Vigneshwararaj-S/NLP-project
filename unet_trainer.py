import os

# 0. HPC CACHE CONFIGURATION
os.environ["HF_HOME"] = "/nobackup/vigsa504/hf_cache"
os.environ["TORCH_HOME"] = "/nobackup/vigsa504/torch_cache"

import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp 

import random

# ==========================================
# 1. PATHS AND HYPERPARAMETERS
# ==========================================
STONE_IMAGES_DIR = "/nobackup/vigsa504/data/chunked_images/stone_images/"
CLEAN_TAMILI_DIR = "/nobackup/vigsa504/data/chunked_images/tamili/"
CLEAN_VATTELUTTU_DIR = "/nobackup/vigsa504/data/chunked_images/vatteluttu/"

OUTPUT_MODEL_PATH = "/nobackup/vigsa504/models/unet_stone_cleaner_fast.pth"

BATCH_SIZE = 4    # Lowered slightly for the small dataset
EPOCHS = 5     # Bumped slightly since epochs will take 2 seconds
LEARNING_RATE = 1e-4
IMG_HEIGHT = 256
IMG_WIDTH = 1024  

# ==========================================
# 2. DATASET MATCHER (Takes a specific list of files)
# ==========================================
class InscriptionDataset(Dataset):
    def __init__(self, file_list, tamili_dir, vatteluttu_dir):
        self.stone_paths = file_list
        self.tamili_dir = tamili_dir
        self.vatteluttu_dir = vatteluttu_dir

    def __len__(self):
        return len(self.stone_paths)

    def __getitem__(self, idx):
        stone_path = self.stone_paths[idx]
        filename = os.path.basename(stone_path)
        clean_filename = filename.replace("stone_", "").replace(".jpg", ".png") 
        
        if "tamili" in clean_filename.lower():
            clean_path = os.path.join(self.tamili_dir, clean_filename)
        elif "vatteluttu" in clean_filename.lower():
            clean_path = os.path.join(self.vatteluttu_dir, clean_filename)
        else:
            raise FileNotFoundError(f"Could not route {clean_filename}")
        
        stone_img = cv2.imread(stone_path, cv2.IMREAD_COLOR)
        stone_img = cv2.cvtColor(stone_img, cv2.COLOR_BGR2RGB)
        clean_mask = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
        
        stone_img = cv2.resize(stone_img, (IMG_WIDTH, IMG_HEIGHT))
        clean_mask = cv2.resize(clean_mask, (IMG_WIDTH, IMG_HEIGHT))
        
        stone_img = stone_img.astype(np.float32) / 255.0
        clean_mask = (clean_mask > 127).astype(np.float32) 
        
        stone_tensor = torch.from_numpy(stone_img).permute(2, 0, 1)
        mask_tensor = torch.from_numpy(clean_mask).unsqueeze(0) 
        
        return stone_tensor, mask_tensor

# ==========================================
# 3. SPLIT LOGIC & TRAINING
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Fast U-Net on: {device}")

    # 1. Gather and sort all files
    all_stones = [os.path.join(STONE_IMAGES_DIR, f) for f in os.listdir(STONE_IMAGES_DIR) if f.endswith(('.png', '.jpg'))]
    
    tamili_files = [f for f in all_stones if "tamili" in f.lower()]
    vatteluttu_files = [f for f in all_stones if "vatteluttu" in f.lower()]

    print(f"Found {len(tamili_files)} Tamili and {len(vatteluttu_files)} Vatteluttu total.")

    # 2. Shuffle and slice randomly!
    print(f"Found {len(tamili_files)} Tamili and {len(vatteluttu_files)} Vatteluttu total.")
    
    # Set a seed so the "random" split is reproducible every time you run it
    random.seed(42) 
    random.shuffle(tamili_files)
    random.shuffle(vatteluttu_files)

    # Now the first 75 and next 25 are completely random selections
    train_files = tamili_files[:500] + vatteluttu_files[:500]
    val_files = tamili_files[500:725] + vatteluttu_files[500:525]

    print(f"🔬 Created Random Training Set: {len(train_files)} images (75/75)")
    print(f"🔬 Created Random Validation Set: {len(val_files)} images (25/25)")

    # 3. Create Datasets
    train_dataset = InscriptionDataset(train_files, CLEAN_TAMILI_DIR, CLEAN_VATTELUTTU_DIR)
    val_dataset = InscriptionDataset(val_files, CLEAN_TAMILI_DIR, CLEAN_VATTELUTTU_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 4. Build Model & Optimizer
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1, activation="sigmoid").to(device)
    criterion = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=False)
    bce_criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)

    # 5. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Train
        for stones, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            stones, masks = stones.to(device), masks.to(device)
            optimizer.zero_grad()
            predictions = model(stones)
            loss = criterion(predictions, masks) + bce_criterion(predictions, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for stones, masks in val_loader:
                stones, masks = stones.to(device), masks.to(device)
                predictions = model(stones)
                loss = criterion(predictions, masks) + bce_criterion(predictions, masks)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")

    print(f"\n✅ Training Complete! Saved to {OUTPUT_MODEL_PATH}")
    torch.save(model.state_dict(), OUTPUT_MODEL_PATH)

if __name__ == "__main__":
    main()