import torch
import torchaudio
import argparse
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, PreTrainedModel
from torch import nn
from safetensors.torch import load_file

# Ensure decoder module is importable
from decoder.decoder import SopranoDecoder
from utils.config_loader import load_config

def load_models(llm_path: str, decoder_path: str, tokenizer_name: str, device: str = 'cuda') -> tuple[PreTrainedModel, nn.Module]:
    if not llm_path or not os.path.exists(llm_path):
        raise FileNotFoundError(f"LLM path invalid or not found: {llm_path}. Please check your config.yaml.")
    if not decoder_path or not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder path invalid or not found: {decoder_path}. Please check your config.yaml.")

    print(f"Loading LLM from {llm_path}...")
    
    # Load LLM Config & Model
    config = AutoConfig.from_pretrained(tokenizer_name)
    llm = AutoModelForCausalLM.from_config(config)
    
    # Load LLM weights
    if llm_path.endswith('.safetensors'):
        state_dict = load_file(llm_path)
        llm.load_state_dict(state_dict)
    else:
        # Fallback for folder/bin
        llm = AutoModelForCausalLM.from_pretrained(llm_path)
    
    llm.to(device).eval()
    
    print(f"Loading Decoder from {decoder_path}...")
    decoder = SopranoDecoder()
    
    # Load Decoder weights
    decoder_state = torch.load(decoder_path, map_location='cpu')
    decoder.load_state_dict(decoder_state)
    decoder.to(device).eval()
    
    return llm, decoder

def generate_audio(text: str, llm: PreTrainedModel, decoder: nn.Module, tokenizer, cfg_inf: dict, device: str = 'cuda', save_path: str = "output.wav"):
    # 1. Format Prompt
    prompt = f"[TEXT]{text}[START]"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"Prompt: {prompt}")
    print("Generating Tokens & Extracting Hidden States...")
    
    # 2. Generate with Hidden States Extraction
    if 'token_type_ids' in inputs: 
        del inputs['token_type_ids']
    
    with torch.no_grad():
        outputs = llm.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs.get('attention_mask'),
            max_new_tokens=cfg_inf["max_new_tokens"], 
            do_sample=True,
            temperature=cfg_inf["temperature"],
            top_k=cfg_inf["top_k"],
            top_p=cfg_inf["top_p"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_hidden_states=True,
            repetition_penalty=cfg_inf["repetition_penalty"]
        )
    
    # 3. Process Hidden States
    hidden_states_list = []
    
    # outputs.hidden_states is a tuple of generated steps.
    # The first element is the Prompt (prefill) hidden states. We skip it.
    for i, step_states in enumerate(outputs.hidden_states):
        # step_states is tuple of layers. Get last layer.
        last_layer_state = step_states[-1][0, -1, :]
        hidden_states_list.append(last_layer_state)
        
    # Concatenate along time dimension
    audio_hidden = torch.stack(hidden_states_list).unsqueeze(0) # (B, T, D)
    audio_hidden = audio_hidden.to(torch.float32)

    num_audio_tokens = audio_hidden.size(1)
    print(f"Generated {num_audio_tokens} audio tokens.")
    
    if num_audio_tokens == 0:
        print("No audio tokens generated! Aborting.")
        return

    # 4. Decode
    decoder_input = audio_hidden.transpose(1, 2)
    print(f"Decoding shape: {decoder_input.shape}...")
    with torch.no_grad():
        audio = decoder(decoder_input)
    
    # 5. Save
    audio = audio.squeeze().cpu() 
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
        
    torchaudio.save(save_path, audio, 32000)
    print(f"Audio saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--out", type=str, default="output.wav", help="Output audio file name")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config("config.yaml")
    cfg_global = config["global"]
    cfg_paths = config["paths"]
    cfg_inf = config["inference"]

    device = cfg_global["device"] if torch.cuda.is_available() else "cpu"
    tokenizer_name = cfg_global.get("tokenizer_name", "ekwek/Soprano-80M")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.eos_token_id = 3
    
    llm_path = cfg_paths["pretrained_llm_path"]
    decoder_path = cfg_paths["pretrained_decoder_path"]
    
    llm, decoder = load_models(llm_path, decoder_path, tokenizer_name, device)
    
    generate_audio(args.text, llm, decoder, tokenizer, cfg_inf, device, args.out)

if __name__ == "__main__":
    main()