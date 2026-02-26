# llm-tts-factory: End-to-End LLM-Backbone TTS Training Framework

llm-tts-factory is a full suite of end-to-end training scripts designed for building an LLM-backbone-based TTS model from scratch. 

Taking inspiration from [Soprano](https://huggingface.co/ekwek/Soprano-1.1-80M), this repository allows you to train a Soprano-style TTS model from the ground up. Because of its architecture, it features an **extra Decoder training step**. Instead of generating audio directly from discrete tokens, this model uses the **hidden states of the LLM** as inputs to generate high-fidelity audio. 

---

## 🏗️ Architecture

The framework is divided into three core stages:

### 1. Codec
- **Description:** Encodes raw audio into discrete units and decodes them back.
- **Current State:** A naive codec encoder and decoder. There is a ton of scope for improvements here (e.g., swapping in RVQ, DAC, or EnCodec).

### 2. LLM Backbone
- **Model:** Qwen-based causal language model.
- **Description:** The core sequence-to-sequence autoregressive model. Takes text and predicts discrete audio representations, passing its hidden states to the decoder.

### 3. Decoder
- **Model:** Vocos-based decoder.
- **Description:** A dedicated vocoder trained with Multi-Resolution STFT and GAN losses to synthesize the final audio waveform directly from the LLM's continuous hidden states.
- **Training strategy:** Trained in multiple stages, first to nail reconstruction and then the perception quality using GAN losses.

---

## 📦 Installation & Setup

This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management. We target Python 3.12 and PyTorch with CUDA 12.8 support by default.

1. **Install `uv`** (if you haven't already):
   ```bash
   pip install uv

```

1. **Initialize the Virtual Environment & Install Dependencies**:

```bash
# Create a seeded Python 3.12 environment
uv venv --python 3.12 --seed

# Activate the environment (Windows)
.venv\Scripts\activate
# Or on Linux/macOS: source .venv/bin/activate

# Sync all dependencies (automatically pulls CUDA 12.8 PyTorch wheels)
uv sync

```

## ⚙️ Configuration

Say goodbye to messy command-line arguments! The entire framework is centrally managed by the `config.yaml` file located in the root directory.

Before running any scripts, open `config.yaml` to set your:

- Dataset and checkpoint paths (relative or absolute).
- Training hyperparameters (batch size, learning rate, max steps, etc.).
- Global settings (device selection, random seed, Weights & Biases logging).

---

## 🪟 Windows Compatibility & Audio Setup

Audio library bindings (`torchaudio`, `torchcodec`) can be problematic and crash-prone on Windows. This framework includes a custom OS-aware `AudioPipeline` (`utils/audio_utils.py`) that detects your operating system.

If you are on Windows, the pipeline will automatically use **ffmpeg** to safely decode, convert to mono, and resample audio into raw PCM float32 arrays without requiring complex C++ bindings.

**Windows Requirements:**

- Ensure `ffmpeg` is added to your system `PATH`, **OR**
- Place the `ffmpeg.exe` binary directly in your project folder at: `./tools/ffmpeg/ffmpeg.exe`.
- *Fallback:* If `ffmpeg` is not found, the pipeline will attempt to fallback to `soundfile` and `scipy`, but `ffmpeg` is highly recommended for speed and format compatibility.

*(Linux users: The pipeline will natively use torchaudio as intended.)*

---

## 🚀 Data Preparation, Training & Inference

Make sure your paths are set in `config.yaml`, then execute the steps in order:

### 0. Data Preparation

Convert your LJSpeech-formatted dataset into audio tokens:

```bash
python generate_dataset.py
# OR
python generate_dataset_from_lists.py

```

### 1. Codec Stage

Train the audio codec encoder and decoder:

```bash
python codec_train.py

```

### 2. LLM Stage

Train the Qwen-based causal LLM to learn the mapping from text to audio representations:

```bash
python train_llm.py

```

### 3. Decoder Stage

Freeze the LLM and train the Vocos decoder to reconstruct the audio from the LLM's hidden states:

```bash
python train_decoder.py

```

### 4. Inference

Run end-to-end inference. (Text is passed via CLI, but generation params like temperature and paths are pulled from `config.yaml`):

```bash
python simple_inference.py --text "hello, my name is soma siddhartha" --out simple_inf_out.wav

```