import os
import sys
# Force evaluate metrics to download to the nobackup folder
os.environ["HF_HOME"] = "/nobackup/vigsa504/hf_cache"

import json
import re
import evaluate
import unicodedata
from master_agent import run_vision_expert 

# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
TEST_DATASET_PATH = "/home/vigsa504/NLP project/test_dataset.jsonl" 
# ⚠️ Point this directly to your noisy stone images!
STONE_DIR = "/nobackup/vigsa504/data/chunked_images/stone_images/"

def prepare_for_metrics(text):
    """Prepares continuous Tamil text for character-level BLEU/ROUGE."""
    if not text: return ""
    text = unicodedata.normalize('NFC', text)
    text = "".join(text.split())
    # Add spaces between every character
    return " ".join(list(text))

def main():
    print(f"Loading test dataset and routing paths to noisy stones in {STONE_DIR}...")
    image_paths = []
    ground_truths = []
    missing_stones = 0
    
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # --- FIX 1: ROUTE TO STONES ---
            old_path = data["image_path"]
            clean_filename = re.split(r'[\\/]', old_path)[-1] 
            
            # Convert "chunk_1.png" to "stone_chunk_1.jpg"
            clean_base = os.path.splitext(clean_filename)[0]
            stone_filename = f"stone_{clean_base}.jpg"
            stone_path = os.path.join(STONE_DIR, stone_filename)
            
            # Fallback just in case some are saved as .png
            if not os.path.exists(stone_path):
                stone_filename_png = f"stone_{clean_base}.png"
                stone_path = os.path.join(STONE_DIR, stone_filename_png)

            if not os.path.exists(stone_path):
                missing_stones += 1
                continue # Skip if we can't find the matching stone
                
            image_paths.append(stone_path)
            
            # --- FIX 2: EXTRACT GROUND TRUTH ---
            assistant_content = data["messages"][1]["content"]
            if isinstance(assistant_content, list):
                gt_text = "".join([item.get("text", "") for item in assistant_content if item.get("type") == "text"])
            else:
                gt_text = str(assistant_content)
                
            ground_truths.append(gt_text.strip())
            
    if missing_stones > 0:
        print(f"⚠️ Warning: Could not find stone equivalents for {missing_stones} images. They were skipped.")

     
    image_paths, ground_truths = image_paths[:50], ground_truths[:50]

    print(f"\nEvaluating 3B Vision Model on {len(image_paths)} NOISY STONE images...")
    predictions = run_vision_expert(image_paths)

    # 1. Prepare character-level strings
    char_predictions = [prepare_for_metrics(p) for p in predictions]
    char_ground_truths = [prepare_for_metrics(gt) for gt in ground_truths]

    # 2. Structure references
    references_bleu = [[gt] for gt in char_ground_truths] 
    references_rouge = char_ground_truths

    # 3. Calculate Metrics
    print("\n📊 CALCULATING CHARACTER-LEVEL METRICS ON NOISY DATA...")
    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    bleu_score = sacrebleu.compute(predictions=char_predictions, references=references_bleu)
    rouge_score = rouge.compute(
        predictions=char_predictions, 
        references=references_rouge,
        tokenizer=lambda x: x.split() 
    )

    print("\n" + "="*45)
    print(" 🪨 NOISY STONE VISION METRICS (3B MODEL) 🪨")
    print("="*45)
    print(f"BLEU Score:   {bleu_score['score']:.2f}  ")
    print(f"ROUGE-1:      {rouge_score['rouge1']:.4f}")
    print(f"ROUGE-2:      {rouge_score['rouge2']:.4f}")
    print(f"ROUGE-L:      {rouge_score['rougeL']:.4f}")
    print("="*45)
    
    # Save log to see exactly how the model hallucinated on the noise
    results_log = [
        {"image": img, "prediction": pred, "reference": ref}
        for img, pred, ref in zip(image_paths, predictions, ground_truths)
    ]
    with open("vision_noisy_stone_results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()