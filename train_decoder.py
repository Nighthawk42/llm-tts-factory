"""
Training script for Soprano Decoder (Vocos).
Freezes LLM and trains Decoder with GAN loss.
"""
import random
import time
import os
import wandb
import matplotlib.pyplot as plt

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file

from dataset_e2e import AudioDataset, SAMPLES_PER_TOKEN
from decoder.decoder import SopranoDecoder
from decoder.discriminator import Discriminator
from decoder.losses import MelSpectrogramWrapper, feature_matching_loss, discriminator_loss, generator_loss, MultiResolutionSTFTLoss

from config_loader import load_config

# Global Tokenizer for collate function
tokenizer = AutoTokenizer.from_pretrained('ekwek/Soprano-80M')
tokenizer.padding_side = 'right' # Essential for training!

def worker_seed_init(_):
    worker_seed = torch.initial_seed() % (2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_lr(it, max_lr, min_lr, warmup_steps, cooldown_steps, max_steps): # WSD schedule
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it < max_steps - cooldown_steps:
        return max_lr
    return min_lr + (max_lr - min_lr) * ((max_steps - it) / cooldown_steps)

def collate_pack(batch_in):
    # batch_in is list of (text, wav)
    texts = [x[0] for x in batch_in]
    wavs = [x[1] for x in batch_in]
    aud_token_lens = [x[2] for x in batch_in]
    
    # We need to process each sample to align audio
    # Since lengths vary, we process list then pad
    batch_tokens_list = []
    batch_audio_list = []
    
    for i in range(len(texts)):
        # Get raw tokens without padding for alignment logic
        raw_tokens = tokenizer(texts[i], padding=False, truncation=False)['input_ids']
        tokens = torch.tensor(raw_tokens, dtype=torch.long)
        
        wav = wavs[i]
        num_aud_tokens = aud_token_lens[i]
        aligned_audio = torch.zeros(num_aud_tokens * SAMPLES_PER_TOKEN, dtype=torch.float32)
        wav_ptr = 0
        is_audio = (tokens > 3) & (tokens <= 8003)
        audio_indices = torch.where(is_audio)[0]
        
        assert len(audio_indices) == num_aud_tokens, f"Audio token count mismatch: {len(audio_indices)} vs {num_aud_tokens}"

        for pos, idx in enumerate(audio_indices):
            if wav_ptr + SAMPLES_PER_TOKEN <= wav.size(0):
                aligned_audio[pos*SAMPLES_PER_TOKEN : (pos+1)*SAMPLES_PER_TOKEN] = wav[wav_ptr : wav_ptr+SAMPLES_PER_TOKEN]
                wav_ptr += SAMPLES_PER_TOKEN
            else:
                break
        
        batch_tokens_list.append(tokens)
        batch_audio_list.append(aligned_audio)

    # Pad Tokens
    batch_tokens = torch.nn.utils.rnn.pad_sequence(batch_tokens_list, batch_first=True, padding_value=0)
    
    # Pad Audio
    batch_audio = torch.nn.utils.rnn.pad_sequence(batch_audio_list, batch_first=True, padding_value=0.0)
    
    x = batch_tokens[:, :-1]
    y = batch_tokens[:, 1:]
    
    # Calculate max seq len of x
    max_len_x = x.size(1)
    gt_audio = batch_audio[:, :max_len_x * SAMPLES_PER_TOKEN]

    # Create Audio Mask (True where token is audio)
    audio_mask = (y > 3) & (y <= 8003) 

    return x, y, gt_audio, audio_mask


if __name__ == '__main__':
    # ------------------
    # Load Configuration
    # ------------------
    config = load_config("config.yaml")
    cfg_global = config["global"]
    cfg_paths = config["paths"]
    cfg_decoder = config["decoder"]

    device = cfg_global["device"]
    seed = cfg_global["seed"]
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    # Setup directories
    train_dataset_path = os.path.join(cfg_paths["dataset_root"], "train.json")
    val_dataset_path = os.path.join(cfg_paths["dataset_root"], "val.json")
    save_path = os.path.join(cfg_paths["save_dir"], "decoder")
    os.makedirs(save_path, exist_ok=True)
    print(f"Save Path: {save_path}")

    if cfg_global["use_wandb"]:
        wandb.init(project=cfg_global["wandb_project"], config=config)

    # Initialize Mel Spectrogram Wrapper dynamically
    mel_fn = MelSpectrogramWrapper().to(device)

    # ------------------
    # Hyperparameters
    # ------------------
    max_steps = cfg_decoder["max_steps"]
    max_lr = float(cfg_decoder["max_lr"])
    min_lr = cfg_decoder["min_lr_ratio"] * max_lr
    warmup_steps = int(max_steps * cfg_decoder["warmup_ratio"])
    cooldown_steps = int(max_steps * cfg_decoder["cooldown_ratio"])
    
    batch_size = cfg_decoder["batch_size"]
    segment_size_samples = cfg_decoder["segment_size_samples"]
    val_freq = cfg_decoder["val_freq"]
    save_freq = cfg_decoder["save_freq"]
    betas = tuple(cfg_decoder["betas"])
    weight_decay = cfg_decoder["weight_decay"]
    start_step = cfg_decoder.get("start_step", 0)

    # Loss Weights
    lambda_mel = cfg_decoder["lambda_mel"]
    lambda_fm = cfg_decoder["lambda_fm"]
    lambda_gen = cfg_decoder["lambda_gen"]
    lambda_stft = cfg_decoder["lambda_stft"]

    # ------------------
    # 1. Load LLM and Freeze
    # ------------------
    print("Loading LLM...")
    llm_config = AutoConfig.from_pretrained('ekwek/Soprano-80M')
    model = AutoModelForCausalLM.from_config(llm_config)

    pretrained_llm_path = cfg_paths["pretrained_llm_path"]
    if pretrained_llm_path and os.path.exists(pretrained_llm_path):
        print(f"Loading custom LLM checkpoint from {pretrained_llm_path}")
        if pretrained_llm_path.endswith('.safetensors'):
            state_dict = load_file(pretrained_llm_path)
            model.load_state_dict(state_dict)
        else:
            model = AutoModelForCausalLM.from_pretrained(pretrained_llm_path)
    else:
        print("Warning: No pretrained LLM provided for Decoder training. Using random init (not recommended).")

    model.to(torch.bfloat16).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("LLM Frozen.")
    
    # ------------------
    # 2. Load Decoder
    # ------------------
    print("Loading Decoder...")
    decoder = SopranoDecoder()
    pretrained_decoder_path = cfg_paths["pretrained_decoder_path"]
    
    if pretrained_decoder_path and os.path.exists(pretrained_decoder_path):
        print(f"Loading custom Decoder checkpoint from {pretrained_decoder_path}")
        decoder.load_state_dict(torch.load(pretrained_decoder_path, map_location='cpu'))
    else:
        print("Training Decoder from scratch.")
        
    decoder.to(device)
    decoder.train() 
    print("Decoder loaded.")

    # Initialize MR-STFT Loss
    mr_stft = MultiResolutionSTFTLoss().to(device)

    # ------------------
    # 3. Load Discriminator
    # ------------------
    discriminator = None
    if cfg_decoder["use_discriminator"]:
        print("Initializing Discriminator...")
        discriminator = Discriminator()
        pretrained_disc_path = cfg_paths["pretrained_discriminator_path"]
        
        if pretrained_disc_path and os.path.exists(pretrained_disc_path):
            print(f"Loading custom Discriminator checkpoint from {pretrained_disc_path}")
            discriminator.load_state_dict(torch.load(pretrained_disc_path, map_location='cpu'))
        
        discriminator.to(device)
        discriminator.train()
    else:
        print("Training WITHOUT Discriminator (Reconstruction only).")

    # ------------------
    # 4. Dataset Setup
    # ------------------
    dataset = AudioDataset(train_dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg_global["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )
    dataloader_it = iter(dataloader)

    val_dataset = AudioDataset(val_dataset_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=max(1, batch_size // 4), # Reduce val batch size to prevent OOM
        shuffle=False,
        num_workers=max(1, cfg_global["num_workers"] // 2),
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=collate_pack,
    )
    val_dataloader_it = iter(val_dataloader)

    # ------------------
    # 5. Optimizers
    # ------------------
    opt_g = torch.optim.AdamW(decoder.parameters(), max_lr, betas=betas, weight_decay=weight_decay)
    opt_d = None
    if cfg_decoder["use_discriminator"]:
        opt_d = torch.optim.AdamW(discriminator.parameters(), max_lr, betas=betas, weight_decay=weight_decay)

    # ------------------
    # Training Loop
    # ------------------
    pbar = tqdm(range(start_step + 1, max_steps + 1), ncols=200, dynamic_ncols=True)
    
    for step in pbar:
        start = time.time()
        
        # Get Data
        try:
            batch_data = next(dataloader_it)
            if batch_data[0] is None: 
                dataloader_it = iter(dataloader)
                batch_data = next(dataloader_it)
            x, y, gt_audio, audio_mask = batch_data 
        except StopIteration:
            dataloader_it = iter(dataloader)
            batch_data = next(dataloader_it)
            x, y, gt_audio, audio_mask = batch_data
            
        x, y = x.to(device), y.to(device)
        gt_audio = gt_audio.to(device) # (B, T_audio_samples)
        audio_mask = audio_mask.to(device)

        # Forward LLM (No Grad)
        with torch.no_grad():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                outputs = model(x, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1] # (B, T_total, D)
                hidden_states = hidden_states.to(torch.float32)

        # GATHER AUDIO LATENTS Logic
        gathered_states_list = []
        for b_idx in range(hidden_states.size(0)):
            mask = audio_mask[b_idx]
            valid_states = hidden_states[b_idx][mask]
            gathered_states_list.append(valid_states)

        decoder_in_padded = torch.nn.utils.rnn.pad_sequence(gathered_states_list, batch_first=True, padding_value=0.0)
        
        bsz = decoder_in_padded.size(0)
        max_aud_len = decoder_in_padded.size(1)
        audio_loss_mask = torch.zeros((bsz, max_aud_len), dtype=torch.bool, device=device)
        for b_idx in range(bsz):
            length = gathered_states_list[b_idx].size(0)
            audio_loss_mask[b_idx, :length] = True

        # ---------------------
        # Generator Forward 
        # ---------------------
        decoder_in = decoder_in_padded.transpose(1, 2) # (B, C, T)
        fake_audio = decoder(decoder_in) # (B, 1, T_audio_gen)
        if fake_audio.size(1) == 1: fake_audio = fake_audio.squeeze(1)
        
        min_len = min(fake_audio.size(1), gt_audio.size(1))
        fake_audio = fake_audio[:, :min_len]
        real_audio = gt_audio[:, :min_len]

        # ---------------------
        # Train Discriminator
        # ---------------------
        d_loss_item = 0.0
        if cfg_decoder["use_discriminator"]:
            opt_d.zero_grad()
            
            # --- Random Cropping Logic ---
            real_crop_list = []
            fake_crop_list = []
            
            for b_idx in range(bsz):
                valid_len = gathered_states_list[b_idx].size(0) * SAMPLES_PER_TOKEN
                valid_len = min(valid_len, min_len) 
                
                if valid_len <= segment_size_samples:
                    pad_len = segment_size_samples - valid_len
                    r_c = torch.nn.functional.pad(real_audio[b_idx, :valid_len], (0, pad_len))
                    f_c = torch.nn.functional.pad(fake_audio[b_idx, :valid_len], (0, pad_len))
                else:
                    start_idx = random.randint(0, valid_len - segment_size_samples)
                    r_c = real_audio[b_idx, start_idx : start_idx + segment_size_samples]
                    f_c = fake_audio[b_idx, start_idx : start_idx + segment_size_samples]
                
                real_crop_list.append(r_c)
                fake_crop_list.append(f_c)
            
            real_crops = torch.stack(real_crop_list).unsqueeze(1) 
            fake_crops = torch.stack(fake_crop_list).unsqueeze(1).detach() 

            y_d_rs, y_d_gs, _, _ = discriminator(real_crops, fake_crops)
            d_loss, _, _ = discriminator_loss(y_d_rs, y_d_gs)
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
            opt_d.step()
            d_loss_item = d_loss.item()
        
        # ---------------------
        # Train Generator
        # ---------------------
        opt_g.zero_grad()
        
        # We need "fake_crops_g" (with grad) for generator loss
        real_crop_list_g = []
        fake_crop_list_g = []
        if cfg_decoder["use_discriminator"]:
             for b_idx in range(bsz):
                valid_len = gathered_states_list[b_idx].size(0) * SAMPLES_PER_TOKEN
                valid_len = min(valid_len, min_len)
                
                if valid_len <= segment_size_samples:
                    pad_len = segment_size_samples - valid_len
                    r_c = torch.nn.functional.pad(real_audio[b_idx, :valid_len], (0, pad_len))
                    f_c = torch.nn.functional.pad(fake_audio[b_idx, :valid_len], (0, pad_len))
                else:
                    start_idx = random.randint(0, valid_len - segment_size_samples)
                    r_c = real_audio[b_idx, start_idx : start_idx + segment_size_samples]
                    f_c = fake_audio[b_idx, start_idx : start_idx + segment_size_samples]
                
                real_crop_list_g.append(r_c)
                fake_crop_list_g.append(f_c)
            
             real_crops_g = torch.stack(real_crop_list_g).unsqueeze(1)
             fake_crops_g = torch.stack(fake_crop_list_g).unsqueeze(1)

        # Mel Loss
        frames_per_token = SAMPLES_PER_TOKEN // 512
        mel_mask = audio_loss_mask.repeat_interleave(frames_per_token, dim=1)
        
        pred_mel = mel_fn(fake_audio)
        gt_mel = mel_fn(real_audio)
        
        min_mel_len = min(pred_mel.size(2), gt_mel.size(2), mel_mask.size(1))
        pred_mel = pred_mel[:, :, :min_mel_len]
        gt_mel = gt_mel[:, :, :min_mel_len]
        mel_mask = mel_mask[:, :min_mel_len]
        
        loss_mel_raw = torch.nn.functional.l1_loss(pred_mel, gt_mel, reduction='none')
        loss_mel = (loss_mel_raw * mel_mask.unsqueeze(1)).sum() / (mel_mask.sum() * pred_mel.size(1) + 1e-6)
        
        # Multi-Resolution STFT Loss
        sample_mask = audio_loss_mask.repeat_interleave(SAMPLES_PER_TOKEN, dim=1)
        sample_mask = sample_mask[:, :min_len]
        
        sc_loss, mag_loss = mr_stft(fake_audio * sample_mask, real_audio * sample_mask)
        
        loss_fm = torch.tensor(0.0, device=device)
        loss_gen = torch.tensor(0.0, device=device)
        
        if cfg_decoder["use_discriminator"]:
            y_d_rs, y_d_gs, fmap_rs, fmap_gs = discriminator(real_crops_g, fake_crops_g)
            loss_fm = feature_matching_loss(fmap_rs, fmap_gs)
            loss_gen, _ = generator_loss(y_d_gs)
        
        total_loss_g = (lambda_mel * loss_mel) + (lambda_gen * loss_gen) + (lambda_fm * loss_fm) + (lambda_stft * (sc_loss + mag_loss))
        
        total_loss_g.backward()
        norm_g = torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        
        # LR Update
        lr = get_lr(step, max_lr, min_lr, warmup_steps, cooldown_steps, max_steps)
        for param_group in opt_g.param_groups: param_group['lr'] = lr
        if cfg_decoder["use_discriminator"]:
            for param_group in opt_d.param_groups: param_group['lr'] = lr / 2
            
        opt_g.step()

        end = time.time()
        dt = (end-start)*1000
        
        tqdm_log = f'mel: {loss_mel.item():.3f} | gen: {loss_gen.item():.3f} | sc: {sc_loss.item():.3f} | mag: {mag_loss.item():.3f} | fm: {loss_fm.item():.3f} | d: {d_loss_item:.3f} | lr: {lr:.2e} | time: {dt:.2f} ms'
        pbar.set_description(tqdm_log)

        # WandB Logging
        log_dict = {
            "train/loss_mel": loss_mel.item(),
            "train/loss_gen": loss_gen.item(),
            "train/loss_fm": loss_fm.item(),
            "train/loss_d": d_loss_item,
            "train/lr": lr,
            "train/total_loss_g": total_loss_g.item(),
            "train/loss_sc": sc_loss.item(),
            "train/loss_mag": mag_loss.item()
        }
        
        # ---------------------
        # Validation Loop
        # ---------------------
        if step % val_freq == 0:
            decoder.eval()
            if discriminator: discriminator.eval()
            
            val_mel_loss_accum = 0.0
            val_gen_loss_accum = 0.0
            val_fm_loss_accum = 0.0
            val_d_loss_accum = 0.0
            val_sc_loss_accum = 0.0
            val_mag_loss_accum = 0.0
            val_steps = 10 
            
            with torch.no_grad():
                for _ in range(val_steps):
                    try:
                        val_batch = next(val_dataloader_it)
                    except StopIteration:
                        val_dataloader_it = iter(val_dataloader)
                        val_batch = next(val_dataloader_it)
                        
                    vx, vy, vgt_audio, vaudio_mask = val_batch
                    vx, vy = vx.to(device), vy.to(device)
                    vgt_audio = vgt_audio.to(device)
                    vaudio_mask = vaudio_mask.to(device)
                    
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        voutputs = model(vx, output_hidden_states=True)
                        v_hidden = voutputs.hidden_states[-1].to(torch.float32)
                    
                    v_gathered_states_list = []
                    for b_idx in range(v_hidden.size(0)):
                        mask = vaudio_mask[b_idx]
                        v_valid_states = v_hidden[b_idx][mask]
                        v_gathered_states_list.append(v_valid_states)
                    
                    v_in_padded = torch.nn.utils.rnn.pad_sequence(v_gathered_states_list, batch_first=True, padding_value=0.0)
                    
                    v_bsz = v_in_padded.size(0)
                    v_max_aud_len = v_in_padded.size(1)
                    v_audio_loss_mask = torch.zeros((v_bsz, v_max_aud_len), dtype=torch.bool, device=device)
                    for b_idx in range(v_bsz):
                        length = v_gathered_states_list[b_idx].size(0)
                        v_audio_loss_mask[b_idx, :length] = True

                    v_in = v_in_padded.transpose(1, 2)
                    v_fake_audio = decoder(v_in)
                    if v_fake_audio.size(1) == 1: v_fake_audio = v_fake_audio.squeeze(1)
                    
                    min_len_v = min(v_fake_audio.size(1), vgt_audio.size(1))
                    v_fake_audio = v_fake_audio[:, :min_len_v]
                    v_real_audio = vgt_audio[:, :min_len_v]
                    
                    frames_per_token_v = SAMPLES_PER_TOKEN // 512
                    v_mel_mask = v_audio_loss_mask.repeat_interleave(frames_per_token_v, dim=1)
                    
                    v_pred_mel = mel_fn(v_fake_audio)
                    v_gt_mel = mel_fn(v_real_audio)
                    
                    min_mel_len_v = min(v_pred_mel.size(2), v_gt_mel.size(2), v_mel_mask.size(1))
                    v_pred_mel = v_pred_mel[:, :, :min_mel_len_v]
                    v_gt_mel = v_gt_mel[:, :, :min_mel_len_v]
                    v_mel_mask = v_mel_mask[:, :min_mel_len_v]
                    
                    v_mel_loss_raw = torch.nn.functional.l1_loss(v_pred_mel, v_gt_mel, reduction='none')
                    v_mel_loss = (v_mel_loss_raw * v_mel_mask.unsqueeze(1)).sum() / (v_mel_mask.sum() * v_pred_mel.size(1) + 1e-6)
                    val_mel_loss_accum += v_mel_loss.item()
                    
                    v_sample_mask = v_audio_loss_mask.repeat_interleave(SAMPLES_PER_TOKEN, dim=1)[:, :min_len_v]
                    v_sc_loss, v_mag_loss = mr_stft(v_fake_audio * v_sample_mask, v_real_audio * v_sample_mask)
                    val_sc_loss_accum += v_sc_loss.item()
                    val_mag_loss_accum += v_mag_loss.item()
                    
                    if cfg_decoder["use_discriminator"]:
                        v_real_crop_list = []
                        v_fake_crop_list = []
                        v_min_len = min(v_fake_audio.size(1), v_real_audio.size(1))
                        
                        for b_idx in range(v_bsz):
                            v_valid_len = v_gathered_states_list[b_idx].size(0) * SAMPLES_PER_TOKEN
                            v_valid_len = min(v_valid_len, v_min_len)
                            
                            if v_valid_len <= segment_size_samples:
                                v_pad_len = segment_size_samples - v_valid_len
                                vr_c = torch.nn.functional.pad(v_real_audio[b_idx, :v_valid_len], (0, v_pad_len))
                                vf_c = torch.nn.functional.pad(v_fake_audio[b_idx, :v_valid_len], (0, v_pad_len))
                            else:
                                v_start_idx = random.randint(0, v_valid_len - segment_size_samples)
                                vr_c = v_real_audio[b_idx, v_start_idx : v_start_idx + segment_size_samples]
                                vf_c = v_fake_audio[b_idx, v_start_idx : v_start_idx + segment_size_samples]
                            
                            v_real_crop_list.append(vr_c)
                            v_fake_crop_list.append(vf_c)
                        
                        v_real_crops = torch.stack(v_real_crop_list).unsqueeze(1)
                        v_fake_crops = torch.stack(v_fake_crop_list).unsqueeze(1)

                        vy_d_rs, vy_d_gs, vfmap_rs, vfmap_gs = discriminator(v_real_crops, v_fake_crops)
                        v_fm_loss = feature_matching_loss(vfmap_rs, vfmap_gs)
                        v_gen_loss, _ = generator_loss(vy_d_gs)
                        v_d_loss, _, _ = discriminator_loss(vy_d_rs, vy_d_gs)
                        
                        val_gen_loss_accum += v_gen_loss.item()
                        val_fm_loss_accum += v_fm_loss.item()
                        val_d_loss_accum += v_d_loss.item()

            # Average metrics
            val_log = {
                "val/loss_mel": val_mel_loss_accum / val_steps,
                "val/loss_gen": val_gen_loss_accum / val_steps,
                "val/loss_fm": val_fm_loss_accum / val_steps,
                "val/loss_d": val_d_loss_accum / val_steps,
                "val/loss_sc": val_sc_loss_accum / val_steps,
                "val/loss_mag": val_mag_loss_accum / val_steps
            }
            log_dict.update(val_log)

            # Generate Mel Images (from last val batch)
            if cfg_global["use_wandb"]:
                gen_mel = mel_fn(v_fake_audio[0:1]).squeeze(0).cpu().numpy()
                gt_mel = mel_fn(v_real_audio[0:1]).squeeze(0).cpu().numpy()
                
                fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                ax[0].imshow(gt_mel, aspect='auto', origin='lower')
                ax[0].set_title("Ground Truth Mel")
                ax[1].imshow(gen_mel, aspect='auto', origin='lower')
                ax[1].set_title("Generated Mel (Val)")
                plt.tight_layout()
                
                log_dict["val/mel_spectrograms"] = wandb.Image(fig)
                plt.close(fig)
            
            decoder.train()
            if discriminator: discriminator.train()

        # Save Checkpoint
        if step > 0 and step % save_freq == 0:
            print(f"\nSaving checkpoint at step {step} to {save_path}...")
            ckpt_name_dec = f"decoder_step_{step}.pth"
            ckpt_name_disc = f"discriminator_step_{step}.pth"
            torch.save(decoder.state_dict(), os.path.join(save_path, ckpt_name_dec))
            if discriminator:
                torch.save(discriminator.state_dict(), os.path.join(save_path, ckpt_name_disc))
        
        if cfg_global["use_wandb"]:
            wandb.log(log_dict, step=step)

    print(f"Training complete. Saving model at {save_path}")
    torch.save(decoder.state_dict(), os.path.join(save_path, "decoder_trained.pth"))
    if discriminator:
        torch.save(discriminator.state_dict(), os.path.join(save_path, "discriminator_trained.pth"))
        
    if cfg_global["use_wandb"]:
        wandb.finish()