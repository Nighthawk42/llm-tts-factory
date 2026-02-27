"""
Converts a dataset in LJSpeech format into audio tokens for Soprano.
This script creates two JSON files for train and test splits in the provided directory.

Usage:
python generate_dataset.py
"""
import pathlib
import json
import os
import random
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from codec.encoder.codec import Encoder

from utils.config_loader import load_config
from utils.audio_utils import AudioPipeline

def load_metadata(input_dir):
    print("Reading metadata...")
    meta_map = {}
    meta_path = input_dir / 'metadata.csv'
    
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}. Did you run sanitize.py?")
    
    with open(meta_path, encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            parts = line.strip().split('|')
            filename = parts[0]
            transcript = parts[-1] 
            meta_map[filename] = transcript
    return meta_map

def main():
    config = load_config("config.yaml")
    cfg_paths = config["paths"]
    cfg_codec = config["codec"]
    cfg_data = config["data_generation"]
    
    input_dir = pathlib.Path(cfg_paths["dataset_root"])
    target_sr = cfg_codec["sample_rate"]
    device = config["global"]["device"] if torch.cuda.is_available() else 'cpu'
    seed = config["global"]["seed"]

    # Load Encoder
    print("Loading Encoder...")
    encoder = Encoder()
    speech_autoencoder_path = cfg_paths["pretrained_codec_path"]
    
    if speech_autoencoder_path and os.path.exists(speech_autoencoder_path):
        print(f"Loading custom weights from {speech_autoencoder_path}...")
        full_ckpt = torch.load(speech_autoencoder_path, map_location='cpu')
        
        encoder_state_dict = {}
        for k, v in full_ckpt.items():
            if k.startswith("encoder."):
                new_k = k.replace("encoder.", "", 1)
                encoder_state_dict[new_k] = v
                
        encoder.load_state_dict(encoder_state_dict)
    else:
        print("No custom codec path found in config. Downloading default Soprano-Encoder from Hugging Face...")
        encoder_path = hf_hub_download(repo_id='ekwek/Soprano-Encoder', filename='encoder.pth')
        encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
        
    encoder.to(device)
    encoder.eval()
    print("Encoder Loaded.")

    meta_map = load_metadata(input_dir)

    print("Encoding audio...")
    dataset = []
    
    # Process all files found in the metadata
    for filename, transcript in tqdm(meta_map.items()):
        wav_path = input_dir / 'wavs' / f'{filename}.wav'
        
        if not wav_path.exists():
            print(f"Warning: {wav_path} not found. Skipping.")
            continue
            
        # Load and Encode with OS-aware pipeline
        try:
            audio, _ = AudioPipeline.load_audio(str(wav_path), target_sr)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            continue
            
        audio = audio.to(device)
        
        with torch.no_grad():
            audio_tokens = encoder(audio) 

        dataset.append([transcript, audio_tokens.squeeze(0).tolist(), str(wav_path.resolve())])

    print("Generating train/test splits...")
    random.seed(seed)
    random.shuffle(dataset)
    num_val = min(int(cfg_data["val_prop"] * len(dataset)) + 1, cfg_data["val_max"])
    
    train_dataset = dataset[num_val:]
    val_dataset = dataset[:num_val]
    
    print(f'# train samples: {len(train_dataset)}')
    print(f'# val samples: {len(val_dataset)}')

    print("Saving datasets...")
    with open(input_dir / 'train.json', 'w', encoding='utf-8') as f:
        json.dump(train_dataset, f, indent=2)
    with open(input_dir / 'val.json', 'w', encoding='utf-8') as f:
        json.dump(val_dataset, f, indent=2)
        
    print("Datasets saved successfully.")

if __name__ == '__main__':
    main()