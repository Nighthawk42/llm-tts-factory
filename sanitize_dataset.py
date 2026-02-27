import re
import shutil
from pathlib import Path
from utils.config_loader import load_config

def clean_text(text):
    """Sanitizes text for TTS training."""
    # Replace weird curly quotes with standard straight quotes
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    
    # Replace em-dashes with standard dashes
    text = text.replace('—', '-')
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text

def main():
    config = load_config("config.yaml")
    dataset_dir = Path(config["paths"]["dataset_root"])
    
    input_csv = dataset_dir / "metadata.csv"
    backup_csv = dataset_dir / "metadata.csv.bak"
    
    if not input_csv.exists():
        print(f"Error: Could not find {input_csv}. Please check your config.yaml.")
        return

    print(f"Reading and sanitizing {input_csv}...")
    
    clean_lines = []
    skipped = 0

    # 1. Read and clean the data in memory first
    with open(input_csv, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Split by pipe
            parts = line.split("|")
            
            if len(parts) < 2:
                print(f"Skipping line {line_num + 1} (not enough columns): {line}")
                skipped += 1
                continue
                
            filename = parts[0].strip()
            
            # Grab the transcript (we take parts[1] so we ignore the duplicate 3rd column if it exists)
            raw_transcript = parts[1]
            
            # Clean the text
            transcript = clean_text(raw_transcript)
            
            # Reformat to strict 2-column: filename|transcript
            clean_lines.append(f"{filename}|{transcript}")

    # 2. Create the backup
    print(f"Creating backup at {backup_csv}...")
    shutil.copy2(input_csv, backup_csv)

    # 3. Overwrite the original file with the clean data
    print(f"Overwriting {input_csv} with clean data...")
    with open(input_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_lines) + "\n")
        
    print(f"Done! Successfully processed {len(clean_lines)} lines. Skipped {skipped} invalid lines.")

if __name__ == "__main__":
    main()