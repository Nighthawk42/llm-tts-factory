"""
Training script for Soprano LLM backbone.

Usage:
python train_llm.py
"""
import os
import random
import time
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file

from dataset import AudioDataset
from utils.config_loader import load_config


def worker_seed_init(_):
    worker_seed = torch.initial_seed() % (2**32-1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lr(it, max_lr, min_lr, warmup_steps, cooldown_steps, max_steps):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it < max_steps - cooldown_steps:
        return max_lr
    return min_lr + (max_lr - min_lr) * ((max_steps - it) / cooldown_steps)


def collate_pack(texts, tokenizer, seq_len, batch_size):
    tokens_batch = tokenizer(texts, padding=False, truncation=False)
    batch = []
    cur_sample, cur_size = [], 0
    for i in range(len(texts)):
        tokens = torch.tensor(tokens_batch['input_ids'][i][:-1], dtype=torch.long)
        cur_size += tokens.size(0)
        cur_sample.append(tokens)
        if cur_size >= seq_len + 1:
            batch.append(torch.cat(cur_sample)[: seq_len + 1])
            cur_sample, cur_size = [], 0
            if len(batch) == batch_size:
                break
    if cur_sample and not batch:
        batch_item = torch.cat(cur_sample + [torch.zeros(seq_len, dtype=torch.long)])[: seq_len + 1]
        batch.append(batch_item)
    if len(batch) < batch_size:
        pad = batch[-1]
        while len(batch) < batch_size:
            batch.append(pad)
    batch = torch.stack(batch)
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


def collate_dynamic(texts, tokenizer):
    tokenized = tokenizer(texts, padding=True, truncation=True, max_length=2048, return_tensors='pt', add_special_tokens=False)
    batch = tokenized['input_ids']
    attn_mask = tokenized['attention_mask']
    
    x = batch[:, :-1]
    y = batch[:, 1:]
    attn_mask = attn_mask[:, :-1]
    
    return x, y, attn_mask


def collate_pack_val(texts, tokenizer, seq_len):
    out = tokenizer(texts, padding=True, truncation=True, max_length=seq_len+1, return_tensors='pt', add_special_tokens=False)
    batch = out['input_ids']
    
    if batch.size(1) < seq_len + 1:
        pad_len = seq_len + 1 - batch.size(1)
        batch = torch.nn.functional.pad(batch, (0, pad_len), value=tokenizer.pad_token_id)
        
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


def compute_loss(x, logits, y, num_steps, mask=None):
    pred = logits.view(-1, logits.size(-1))
    labels = y.reshape(-1)
    loss = torch.nn.functional.cross_entropy(pred, labels, reduction='none')
    
    if mask is not None:
        mask = mask.reshape(-1)
        loss = loss * mask
    
    audio_mask_cond = torch.logical_and(labels >= 3, labels <= 8003)
    if mask is not None:
        audio_mask = audio_mask_cond & (mask > 0)
    else:
        audio_mask = audio_mask_cond
    
    if mask is not None:
        text_mask = (~audio_mask_cond) & (mask > 0)
    else:
        text_mask = ~audio_mask_cond

    audio_mean = loss[audio_mask].mean() if audio_mask.sum() > 0 else torch.tensor(0.0, device=loss.device)
    text_mean = loss[text_mask].mean() if text_mask.sum() > 0 else torch.tensor(0.0, device=loss.device)
    
    acc = (logits.argmax(dim=-1).view(-1) == labels).view(-1)[audio_mask].to(torch.float32).mean()
    if torch.isnan(acc): acc = torch.tensor(0.0, device=loss.device)

    audio_loss = audio_mean / num_steps
    text_loss = text_mean / num_steps
    acc = acc / num_steps
    return audio_loss, text_loss, acc


def evaluate(model, val_dataloader, step, device, use_wandb):
    model.eval()
    val_dataloader_it = iter(val_dataloader)
    with torch.no_grad():
        val_audio_loss_accum = torch.tensor(0.0).to(device)
        val_text_loss_accum = torch.tensor(0.0).to(device)
        val_acc_accum = torch.tensor(0.0).to(device)
        val_loss_steps = len(val_dataloader)
        
        for _ in range(val_loss_steps):
            x, y, attn_mask = next(val_dataloader_it)
            x, y, attn_mask = x.to(device), y.to(device), attn_mask.to(device)
            
            logits = model(x, attention_mask=attn_mask).logits
            audio_loss, text_loss, acc = compute_loss(x, logits, y, val_loss_steps, mask=attn_mask)
            
            val_audio_loss_accum += audio_loss.detach()
            val_text_loss_accum += text_loss.detach()
            val_acc_accum += acc.detach()
            
        print(f"validation text loss: {val_text_loss_accum.item():.4f}\tvalidation audio loss: {val_audio_loss_accum.item():.4f}\tvalidation acc: {val_acc_accum.item():.4f}")
        
        if use_wandb:
            wandb.log({
                "val/text_loss": val_text_loss_accum.item(),
                "val/audio_loss": val_audio_loss_accum.item(),
                "val/acc": val_acc_accum.item()
            }, step=step)
            
    model.train()


def main():
    config = load_config("config.yaml")
    cfg_global = config["global"]
    cfg_paths = config["paths"]
    cfg_llm = config["llm"]

    device = cfg_global["device"]
    seed = cfg_global["seed"]
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    tokenizer_name = cfg_global.get("tokenizer_name", "ekwek/Soprano-80M")
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.set_float32_matmul_precision('high')

    train_dataset_path = os.path.join(cfg_paths["dataset_root"], "train.json")
    val_dataset_path = os.path.join(cfg_paths["dataset_root"], "val.json")
    save_path = os.path.join(cfg_paths["save_dir"], "llm")
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Save Path: {save_path}")

    if cfg_global["use_wandb"]:
        wandb.init(project=cfg_global["wandb_project"], config=config)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.padding_side = 'right'

    max_steps = cfg_llm["max_steps"]
    max_lr = float(cfg_llm["max_lr"])
    min_lr = cfg_llm["min_lr_ratio"] * max_lr
    warmup_steps = int(max_steps * cfg_llm["warmup_ratio"])
    cooldown_steps = int(max_steps * cfg_llm["cooldown_ratio"])
    
    batch_size = cfg_llm["batch_size"]
    grad_accum_steps = cfg_llm["grad_accum_steps"]
    seq_len = cfg_llm["seq_len"]
    val_freq = cfg_llm["val_freq"]
    save_freq = cfg_llm["save_freq"]
    text_factor = cfg_llm["text_factor"]
    betas = tuple(cfg_llm["betas"])
    weight_decay = cfg_llm["weight_decay"]

    if cfg_llm["from_scratch"]:
        print("Initializing model from scratch (random weights)...")
        m_config = AutoConfig.from_pretrained(tokenizer_name)
        model = AutoModelForCausalLM.from_config(m_config)
    else:
        pretrained_path = cfg_paths["pretrained_llm_path"]
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained model weights from {pretrained_path}...")
            m_config = AutoConfig.from_pretrained(tokenizer_name)
            model = AutoModelForCausalLM.from_config(m_config)
            state_dict = load_file(pretrained_path)
            model.load_state_dict(state_dict)
        else:
            print("Loading default pretrained model weights from HF...")
            model = AutoModelForCausalLM.from_pretrained(tokenizer_name)

    model.to(device)
    model.train()

    dataset = AudioDataset(train_dataset_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg_global["num_workers"],
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=lambda texts: collate_dynamic(texts, tokenizer),
    )
    dataloader_it = iter(dataloader)
    
    val_dataset = AudioDataset(val_dataset_path)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, cfg_global["num_workers"] // 2),
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=worker_seed_init,
        collate_fn=lambda texts: collate_dynamic(texts, tokenizer),
    )

    opt = torch.optim.AdamW(model.parameters(), max_lr, betas=betas, weight_decay=weight_decay, fused=True)

    start_step = 1 
    pbar = tqdm(range(start_step, max_steps + 1), ncols=200, dynamic_ncols=True)
    
    for step in pbar:
        start = time.time()
        
        if val_freq > 0 and step != start_step and (step % val_freq == 0 or step == max_steps):
            evaluate(model, val_dataloader, step, device, cfg_global["use_wandb"])
        
        if save_freq > 0 and step % save_freq == 0:
            ckpt_dir = os.path.join(save_path, f"checkpoint-{step}")
            print(f"\nSaving checkpoint to {ckpt_dir}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

        opt.zero_grad()
        audio_loss_accum = 0.0
        text_loss_accum = 0.0
        acc_accum = 0.0
        
        for micro_step in range(grad_accum_steps):
            try:
                x, y, attn_mask = next(dataloader_it)
            except StopIteration:
                dataloader_it = iter(dataloader)
                x, y, attn_mask = next(dataloader_it)
                
            x, y, attn_mask = x.to(device), y.to(device), attn_mask.to(device)

            logits = model(x, attention_mask=attn_mask).logits
            audio_loss, text_loss, acc = compute_loss(x, logits, y, grad_accum_steps, mask=attn_mask)
            
            audio_loss_accum += audio_loss.detach()
            text_loss_accum += text_loss.detach()
            acc_accum += acc.detach()
            
            total_loss = audio_loss + text_factor * text_loss
            total_loss.backward()

        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step, max_lr, min_lr, warmup_steps, cooldown_steps, max_steps)
        
        for param_group in opt.param_groups:
            param_group['lr'] = lr
        opt.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
            
        end = time.time()
        dt = (end - start) * 1000
        tokens_per_second = (batch_size * seq_len * grad_accum_steps) / (end - start)
        
        tqdm_log = f'text loss: {text_loss_accum.item():.3f} | audio loss: {audio_loss_accum.item():.3f} | acc: {acc_accum.item():.4f} | lr: {lr:.2e} | norm: {norm:.3f} | time: {dt:.2f} ms | {tokens_per_second:.2f} t/s'
        pbar.set_description(tqdm_log)
        
        if cfg_global["use_wandb"]:
            wandb.log({
                "train/text_loss": text_loss_accum.item(),
                "train/audio_loss": audio_loss_accum.item(),
                "train/acc": acc_accum.item(),
                "train/lr": lr,
                "train/grad_norm": norm,
                "train/dt": dt,
                "train/tokens_per_sec": tokens_per_second
            }, step=step)

    print(f"\nTraining complete. Saving final model at {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Saving done.")
    
    if cfg_global["use_wandb"]:
        wandb.finish()

if __name__ == '__main__':
    main()