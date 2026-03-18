import os
# Force evaluate metrics to download to the nobackup folder
os.environ["HF_HOME"] = "/nobackup/vigsa504/hf_cache"

import json
import re
import evaluate
from master_agent import run_vision_expert 
import unicodedata

# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
TEST_DATASET_PATH = "/home/vigsa504/NLP project/test_dataset.jsonl" 
VATTELUTTU_DIR = "/nobackup/vigsa504/data/chunked_images/vatteluttu/"
TAMILI_DIR = "/nobackup/vigsa504/data/chunked_images/tamili/"


def prepare_for_metrics(text):
    """Prepares continuous Tamil text for character-level BLEU/ROUGE."""
    if not text: return ""
    # 1. Normalize Unicode (Fixes invisible encoding differences)
    text = unicodedata.normalize('NFC', text)
    # 2. Remove internal newlines and extra whitespace
    text = "".join(text.split())
    # 3. CRITICAL: Add spaces between every character so BLEU/ROUGE 
    # treats each Tamil letter as a 'word'.
    return " ".join(list(text))

def main():
    print("Loading test dataset and fixing paths dynamically...")
    image_paths = []
    ground_truths = []
    
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # --- FIX 1: DYNAMIC PATH ROUTING ---
            old_path = data["image_path"]
            filename = re.split(r'[\\/]', old_path)[-1] # Extract just the filename
            
            if "vatteluttu" in filename.lower():
                new_path = os.path.join(VATTELUTTU_DIR, filename)
            elif "tamili" in filename.lower():
                new_path = os.path.join(TAMILI_DIR, filename)
            else:
                new_path = old_path
                
            image_paths.append(new_path)
            
            # --- FIX 2: EXTRACT RAW STRING FROM DICTIONARY ---
            assistant_content = data["messages"][1]["content"]
            
            # If it's a list of dicts (Multimodal format)
            if isinstance(assistant_content, list):
                gt_text = ""
                for item in assistant_content:
                    if item.get("type") == "text":
                        gt_text += item.get("text", "")
            # If it's already a string
            else:
                gt_text = str(assistant_content)
                
            ground_truths.append(gt_text.strip())
            
    # ⚠️ SMOKE TEST: RUN 50 IMAGES FIRST! 
    image_paths, ground_truths = image_paths[:50], ground_truths[:50]

    print(f"\nEvaluating Vision Agent on {len(image_paths)} images...")
    predictions = run_vision_expert(image_paths)

    # ... after running predictions ...

    # 1. Prepare character-level strings
    char_predictions = [prepare_for_metrics(p) for p in predictions]
    char_ground_truths = [prepare_for_metrics(gt) for gt in ground_truths]

    # 2. Use the char-spaced versions for references
    # Note: sacrebleu needs a list of lists, rouge needs a list of strings
    references_bleu = [[gt] for gt in char_ground_truths] 
    references_rouge = char_ground_truths

    # 3. Calculate Metrics using the CHAR versions
    print("\n📊 CALCULATING CHARACTER-LEVEL ROUGE AND BLEU SCORES...")
    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    # CRITICAL: predictions=char_predictions must be used here!
    bleu_score = sacrebleu.compute(predictions=char_predictions, references=references_bleu)
    rouge_score = rouge.compute(
        predictions=char_predictions, 
        references=references_rouge,
        tokenizer=lambda x: x.split() 
    )

    print("\n" + "="*40)
    print("🏆 VISION MODEL METRICS 🏆")
    print("="*40)
    print(f"BLEU Score:   {bleu_score['score']:.2f}")
    print(f"ROUGE-1:      {rouge_score['rouge1']:.4f}")
    print(f"ROUGE-2:      {rouge_score['rouge2']:.4f}")
    print(f"ROUGE-L:      {rouge_score['rougeL']:.4f}")
    print("="*40)
    
    # 4. Save a log so you can look at the actual outputs for your report
    results_log = [
        {"image": img, "prediction": pred, "reference": ref}
        for img, pred, ref in zip(image_paths, predictions, ground_truths)
    ]
    with open("vision_thesis_results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=4)
    print("Detailed outputs saved to 'vision_thesis_results.json'")

if __name__ == "__main__":
    main()