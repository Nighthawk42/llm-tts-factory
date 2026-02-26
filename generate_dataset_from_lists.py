"""
Converts a dataset in LJSpeech format into audio tokens for Soprano, using pre-defined train/val lists.

Usage:
python generate_dataset_from_lists.py
"""
import pathlib
import json
import os
import torch
from tqdm import tqdm
from encoder.codec import Encoder

from config_loader import load_config
from utils.audio_utils import AudioPipeline

def load_metadata(input_dir):
    print("Reading metadata...")
    meta_map = {}
    meta_path = input_dir / 'metadata_orig.csv'
    if not meta_path.exists():
        meta_path = input_dir / 'metadata.csv'
    
    with open(meta_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('|')
            filename = parts[0]
            transcript = parts[-1] 
            meta_map[filename] = transcript
    return meta_map

def process_list(list_file, meta_map, encoder, target_sr, device):
    dataset = []
    print(f"Processing {list_file}...")
    with open(list_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    for line in tqdm(lines):
        path_obj = pathlib.Path(line)
        filename = path_obj.stem # LJxxx
        
        if filename not in meta_map:
            print(f"Warning: {filename} not found in metadata. Skipping.")
            continue
            
        transcript = meta_map[filename]
        wav_path = str(path_obj)
        
        # Load and Encode with OS-aware pipeline
        try:
            audio, _ = AudioPipeline.load_audio(wav_path, target_sr)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue
        
        audio = audio.to(device)
        
        with torch.no_grad():
            audio_tokens = encoder(audio) 

        dataset.append([transcript, audio_tokens.squeeze(0).tolist(), wav_path])
        
    return dataset

def main():
    config = load_config("config.yaml")
    cfg_paths = config["paths"]
    cfg_codec = config["codec"]
    
    input_dir = pathlib.Path(cfg_paths["dataset_root"])
    
    # Save lists into the configured save_dir
    output_dir = pathlib.Path(cfg_paths["save_dir"]) / "dataset_lists"
    os.makedirs(output_dir, exist_ok=True)
    
    target_sr = cfg_codec["sample_rate"]
    device = config["global"]["device"] if torch.cuda.is_available() else 'cpu'

    # Load Encoder
    print("Loading Encoder...")
    encoder = Encoder()
    speech_autoencoder_path = cfg_paths["pretrained_codec_path"]
    
    if not speech_autoencoder_path or not os.path.exists(speech_autoencoder_path):
        raise FileNotFoundError(f"pretrained_codec_path not found: {speech_autoencoder_path}")
        
    print(f"Loading weights from {speech_autoencoder_path}")
    full_ckpt = torch.load(speech_autoencoder_path, map_location='cpu')
    
    encoder_state_dict = {}
    for k, v in full_ckpt.items():
        if k.startswith("encoder."):
            new_k = k.replace("encoder.", "", 1)
            encoder_state_dict[new_k] = v
            
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)
    encoder.eval()
    print("Encoder Loaded.")

    meta_map = load_metadata(input_dir)

    # Process Train List
    train_list_path = input_dir / 'train_list.txt'
    if train_list_path.exists():
        train_data = process_list(train_list_path, meta_map, encoder, target_sr, device)
        with open(output_dir / 'train.json', 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Saved {len(train_data)} train samples to {output_dir}/train.json")
    else:
        print(f"Error: {train_list_path} not found.")

    # Process Val List
    val_list_path = input_dir / 'val_list.txt'
    if val_list_path.exists():
        val_data = process_list(val_list_path, meta_map, encoder, target_sr, device)
        with open(output_dir / 'val.json', 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Saved {len(val_data)} val samples to {output_dir}/val.json")
    else:
        print(f"Error: {val_list_path} not found.")

if __name__ == '__main__':
    main()