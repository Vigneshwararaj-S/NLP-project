import os
# Force evaluate metrics to download to the nobackup folder
os.environ["HF_HOME"] = "/nobackup/vigsa504/hf_cache"

import json
import evaluate
import unicodedata

# Import the 0.5B NLP expert instead of the Vision expert
from master_agent import run_linguistic_expert 

# ==========================================
# EVALUATION CONFIGURATION
# ==========================================
TEST_DATASET_PATH = "segmenter_test_dataset.jsonl" 

def clean_text_word_level(text):
    """Standardizes text for word-level evaluation."""
    if not text: return ""
    # Normalize Unicode (Fixes invisible encoding differences)
    text = unicodedata.normalize('NFC', str(text))
    # Remove internal newlines and standardizes spaces
    return " ".join(text.split())

def main():
    print("Loading segmentation dataset...")
    inputs = []
    ground_truths = []
    
    with open(TEST_DATASET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            
            # The continuous string to feed the model
            user_msg = data["messages"][1]["content"]
            # The perfectly spaced string to check against
            assistant_msg = data["messages"][2]["content"]
            
            inputs.append(user_msg.strip())
            ground_truths.append(assistant_msg.strip())
            
    # ⚠️ SMOKE TEST: RUN 50 LINES FIRST! 
    inputs, ground_truths = inputs[:50], ground_truths[:50]

    print(f"\nEvaluating NLP Segmenter on {len(inputs)} strings...")
    # Call the 0.5B model from your master agent
    predictions = run_linguistic_expert(inputs)

    # 1. Prepare word-level strings
    clean_predictions = [clean_text_word_level(p) for p in predictions]
    clean_ground_truths = [clean_text_word_level(gt) for gt in ground_truths]

    # 2. Structure references properly
    references_bleu = [[gt] for gt in clean_ground_truths] 
    references_rouge = clean_ground_truths

    # 3. Calculate Metrics
    print("\n📊 CALCULATING WORD-LEVEL ROUGE AND BLEU SCORES...")
    sacrebleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")

    bleu_score = sacrebleu.compute(predictions=clean_predictions, references=references_bleu)
    
    # ROUGE needs the split tokenizer override so it doesn't delete Tamil text
    rouge_score = rouge.compute(
        predictions=clean_predictions, 
        references=references_rouge,
        tokenizer=lambda x: x.split() 
    )

    print("\n" + "="*40)
    print("🏆 SEGMENTATION MODEL METRICS 🏆")
    print("="*40)
    print(f"BLEU Score:   {bleu_score['score']:.2f}")
    print(f"ROUGE-1:      {rouge_score['rouge1']:.4f}")
    print(f"ROUGE-2:      {rouge_score['rouge2']:.4f}")
    print(f"ROUGE-L:      {rouge_score['rougeL']:.4f}")
    print("="*40)
    
    # 4. Save a log to check for weird Sandhi splitting errors
    results_log = [
        {"input": inp, "prediction": pred, "reference": ref}
        for inp, pred, ref in zip(inputs, clean_predictions, clean_ground_truths)
    ]
    with open("segmenter_thesis_results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=4)
    print("Detailed outputs saved to 'segmenter_thesis_results.json'")

if __name__ == "__main__":
    main()