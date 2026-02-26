import torch
from torch.utils.data import Dataset
import os
import json

from utils.audio_utils import AudioPipeline

class LJSpeechDataset(Dataset):
    def __init__(self, root, sample_rate=32000,  mode='train'):
        """
        root: path to LJSpeech-1.1 directory
        """
        self.root = root
        self.sample_rate = sample_rate
        self.mode = mode
        
        mode_json = os.path.join(root, f"{mode}.json")
        if not os.path.exists(mode_json):
            raise FileNotFoundError(f"Dataset JSON not found: {mode_json}")
            
        with open(mode_json, 'r') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text, audio_tokens, wav_path = item

        # Use the robust OS-aware pipeline to load, convert to mono, and resample
        try:
            wav, _ = AudioPipeline.load_audio(wav_path, target_sr=self.sample_rate)
        except Exception as e:
            print(f"Error loading {wav_path}: {e}")
            # Fallback to 1 second of silence to prevent the dataloader from crashing the entire run
            wav = torch.zeros((1, self.sample_rate))

        return wav