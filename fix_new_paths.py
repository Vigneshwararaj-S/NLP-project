import json
import re
import os

# --- PATHS ---
INPUT_JSONL = "vlm_dataset.jsonl"
OUTPUT_JSONL = "/nobackup/vigsa504/data/fixed_dataset.jsonl"

# YOUR TWO NEW TARGET FOLDERS
VATTELUTTU_DIR = "/nobackup/vigsa504/data/chunked_images/vatteluttu/"
TAMILI_DIR = "/nobackup/vigsa504/data/chunked_images/tamili/"

def fix_image_paths():
    print("Reading dataset and routing paths to correct folders...")
    processed_count = 0
    missing_images = 0
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            
            # Dig into the chat template to find the image path
            for message in data.get("messages", []):
                for content in message.get("content", []):
                    if content.get("type") == "image":
                        old_path = content["image"]
                        
                        # Extract just the filename (e.g., 'tamili_001.png')
                        filename = re.split(r'[\\/]', old_path)[-1]
                        
                        # Route to the correct folder based on the filename
                        if "vatteluttu" in filename.lower():
                            new_path = os.path.join(VATTELUTTU_DIR, filename)
                        elif "tamili" in filename.lower():
                            new_path = os.path.join(TAMILI_DIR, filename)
                        else:
                            print(f"⚠️ Warning: Unknown script type for {filename}. Leaving path as is.")
                            new_path = old_path
                            
                        # Safety check: Verify the image actually exists on the cluster
                        if not os.path.exists(new_path):
                            print(f"❌ Missing File: {new_path}")
                            missing_images += 1
                            
                        content["image"] = new_path
            
            # Write the fixed row into the new file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_count += 1
            
    print(f"\n✅ Successfully updated {processed_count} rows!")
    if missing_images > 0:
        print(f"⚠️ WARNING: {missing_images} images could not be found on the disk. Check your folders!")
    else:
        print("✅ All image files verified and found on disk!")
        
    print(f"✅ New dataset saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    fix_image_paths()