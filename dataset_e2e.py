import json
import torch
from torch.utils.data import Dataset
from utils.audio_utils import AudioPipeline

# The codec downsamples audio by a factor of 2048. 
# At 32kHz, 1 token = 2048 raw audio samples.
SAMPLES_PER_TOKEN = 2048

class AudioDataset(Dataset):
    def __init__(self, path, target_sr=32000):
        with open(path, encoding='utf-8') as f:
            self.dataset = json.load(f)
        self.target_sr = target_sr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # The JSON format from generate_dataset.py is:
        # [transcript, audio_tokens (list), audio_path]
        text, audio_tokens, audio_path = self.dataset[idx]
        
        # CRITICAL FIX: We must format the string with the audio tokens physically embedded 
        # so train_decoder.py can tokenize it and find the audio indices to align the waveform!
        formatted_text = f"[TEXT]{text}[START]{''.join(list(map(lambda x: f'[{x}]', audio_tokens)))}[STOP]"
        
        # Use our robust OS-aware pipeline to load the audio
        try:
            wav, _ = AudioPipeline.load_audio(audio_path, target_sr=self.target_sr)
            # Squeeze out the channel dimension so it's a 1D tensor (T,) 
            # as expected by train_decoder.py's alignment logic
            wav = wav.squeeze(0) 
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Fallback to silence to prevent dataloader crashes
            wav = torch.zeros(len(audio_tokens) * SAMPLES_PER_TOKEN)

        # Return the formatted string, the waveform, and the token count
        return formatted_text, wav, len(audio_tokens)