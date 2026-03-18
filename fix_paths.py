import json
import re
import os

# --- PATHS ---
INPUT_JSONL = "final_vlm_dataset.jsonl"
OUTPUT_JSONL = "/nobackup/vigsa504/data/fixed_dataset.jsonl"

# THIS IS THE FOLDER WHERE YOUR EXTRACTED IMAGES ACTUALLY LIVE NOW
NEW_IMAGE_DIR = "/nobackup/vigsa504/data/chunked_images/stone_images/"

def fix_image_paths():
    print("Reading dataset and updating paths...")
    processed_count = 0
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_JSONL, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            data = json.loads(line)
            
            # Dig into the chat template to find the image path
            for message in data.get("messages", []):
                for content in message.get("content", []):
                    if content.get("type") == "image":
                        old_path = content["image"]
                        
                        # Extract just the filename (e.g., 'stone_001.png')
                        # re.split handles both Windows '\' and Mac/Linux '/' slashes
                        filename = re.split(r'[\\/]', old_path)[-1]
                        
                        # Glue the filename to your new server path
                        new_path = os.path.join(NEW_IMAGE_DIR, filename)
                        content["image"] = new_path
            
            # Write the fixed row into the new file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
            processed_count += 1
            
    print(f"✅ Successfully updated {processed_count} rows!")
    print(f"✅ New dataset saved to: {OUTPUT_JSONL}")

if __name__ == "__main__":
    fix_image_paths()