import platform
import os
import subprocess
import numpy as np
import torch
import soundfile as sf
import torchaudio

class AudioPipeline:
    @staticmethod
    def save_audio(file_path, audio_tensor, sample_rate=32000):
        """OS-aware audio saving."""
        system = platform.system()
        
        # Ensure audio is on CPU and is a numpy array for universal saving
        if isinstance(audio_tensor, torch.Tensor):
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = audio_tensor
            
        # Ensure shape is (samples,) or (samples, channels) for soundfile
        if len(audio_np.shape) == 2 and audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)
            
        if system == "Windows":
            # Windows: torchaudio.save is broken without C++ bindings, use soundfile
            sf.write(file_path, audio_np, sample_rate)
        else:
            # Linux: torchaudio works perfectly
            # torchaudio expects shape (channels, samples)
            audio_tensor_save = torch.from_numpy(audio_np).unsqueeze(0).float()
            torchaudio.save(file_path, audio_tensor_save, sample_rate)

    @staticmethod
    def load_audio(file_path, target_sr=32000):
        """OS-aware audio loading and resampling."""
        system = platform.system()
        if system == "Windows":
            return AudioPipeline._load_windows(file_path, target_sr)
        else:
            return AudioPipeline._load_linux(file_path, target_sr)

    @staticmethod
    def _load_linux(file_path, target_sr):
        # Linux handles torchaudio beautifully
        audio, sr = torchaudio.load(file_path)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=target_sr)
        
        # Ensure Mono
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        return audio, target_sr

    @staticmethod
    def _load_windows(file_path, target_sr):
        # Check for local ffmpeg first, then fallback to system PATH
        ffmpeg_cmd = "ffmpeg"
        if os.path.exists("./tools/ffmpeg/ffmpeg.exe"):
            ffmpeg_cmd = "./tools/ffmpeg/ffmpeg.exe"
        
        # Force sample rate, mono channel, and output to raw PCM float32
        command = [
            ffmpeg_cmd,
            "-i", str(file_path),
            "-ac", "1",            # Force Mono
            "-ar", str(target_sr), # Target sample rate
            "-f", "f32le",         # Format: float32 little endian
            "-hide_banner",
            "-loglevel", "error",
            "-"                    # Output to stdout
        ]

        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = process.communicate()
            
            if process.returncode != 0:
                print(f"ffmpeg error for {file_path}, falling back to soundfile. Error: {err.decode('utf-8')}")
                return AudioPipeline._load_windows_fallback(file_path, target_sr)

            # Read raw bytes directly into a numpy array, then to torch tensor
            audio_np = np.frombuffer(out, dtype=np.float32).copy()
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0) # Shape: (1, T)
            return audio_tensor, target_sr

        except FileNotFoundError:
            print(f"ffmpeg not found in PATH or ./tools/ffmpeg/. Falling back to soundfile for {file_path}")
            return AudioPipeline._load_windows_fallback(file_path, target_sr)

    @staticmethod
    def _load_windows_fallback(file_path, target_sr):
        # Soundfile doesn't natively resample, so we rely on scipy
        import scipy.signal
        
        audio_np, sr = sf.read(file_path)
        
        # Convert to mono if needed
        if len(audio_np.shape) > 1:
            audio_np = audio_np.mean(axis=1)
        
        if sr != target_sr:
            num_samples = int(round(len(audio_np) * float(target_sr) / sr))
            audio_np = scipy.signal.resample(audio_np, num_samples)
        
        # Shape to (1, T)
        audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
        return audio_tensor, target_sr