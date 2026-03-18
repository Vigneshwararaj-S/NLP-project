import os
import sys

# Import your two experts from the master agent
from master_agent import run_vision_expert, run_linguistic_expert

# ==========================================
# INPUT
# ==========================================
# ⚠️ Change this to whatever image you want to test!
TEST_IMAGE_PATH = "Ins7_ink (1).jpg" 

def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Error: Could not find image at {TEST_IMAGE_PATH}")
        sys.exit(1)

    print("\n" + "="*50)
    print(" 🚀 STARTING END-TO-END OCR PIPELINE")
    print("="*50)

    # ---------------------------------------------------------
    # STAGE 1: VISION EXPERT (3B Parameter Model)
    # ---------------------------------------------------------
    print(f"\n📸 STAGE 1: Analyzing Image -> {TEST_IMAGE_PATH}")
    # The function expects a list of paths, so we wrap it in []
    vision_results = run_vision_expert([TEST_IMAGE_PATH])
    
    # Extract the single string result
    raw_continuous_text = vision_results[0]
    
    print("\n" + "-"*40)
    print("🧠 VISION MODEL OUTPUT (Raw & Continuous):")
    print("-"*40)
    print(raw_continuous_text)
    print("-"*40)

    # ---------------------------------------------------------
    # STAGE 2: LINGUISTIC EXPERT (0.5B Parameter Segmenter)
    # ---------------------------------------------------------
    print("\n📝 STAGE 2: Applying Sandhi Segmentation...")
    # The function expects a list of strings, so we wrap it in []
    linguistic_results = run_linguistic_expert([raw_continuous_text])
    
    # Extract the final spaced string
    final_spaced_text = linguistic_results[0]

    print("\n" + "="*50)
    print(" ✨ FINAL PIPELINE OUTPUT (Modern Spacing) ✨")
    print("="*50)
    print(final_spaced_text)
    print("="*50 + "\n")

if __name__ == "__main__":
    main()