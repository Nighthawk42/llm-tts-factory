import os
import glob
import subprocess
import argparse
import yaml
from pathlib import Path

def get_latest_checkpoint(base_path, pattern):
    """Finds the latest checkpoint file/folder based on step number."""
    checkpoints = glob.glob(os.path.join(base_path, pattern))
    if not checkpoints:
        return None
    
    # Extract step number and sort
    # For LLM: 'checkpoint-3000' -> 3000
    # For Decoder: 'decoder_step_5000.pth' -> 5000
    try:
        if "checkpoint-" in checkpoints[0]:
            checkpoints.sort(key=lambda x: int(x.split('-')[-1]))
        else:
            import re
            checkpoints.sort(key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[0]))
    except Exception:
        checkpoints.sort()
        
    return checkpoints[-1]

def update_config(config_path, updates):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Deep update logic
    for section, values in updates.items():
        if section in config:
            config[section].update(values)
        else:
            config[section] = values

    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description="Resume training Phase 2 or Phase 3.")
    parser.add_argument("phase", choices=["llm", "decoder"], help="Which phase to resume")
    args = parser.parse_args()

    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    save_dir = config["paths"]["save_dir"]

    if args.phase == "llm":
        ckpt_dir = os.path.join(save_dir, "llm")
        latest = get_latest_checkpoint(ckpt_dir, "checkpoint-*")
        
        if latest:
            print(f"Found latest LLM checkpoint: {latest}")
            # HuggingFace expects the path to the folder or the specific safetensors file
            model_file = os.path.join(latest, "model.safetensors")
            update_config("config.yaml", {
                "paths": {"pretrained_llm_path": model_file},
                "llm": {"from_scratch": False}
            })
            print("Config updated. Resuming LLM training...")
            subprocess.run(["python", "train_llm.py"])
        else:
            print("No LLM checkpoints found to resume from.")

    elif args.phase == "decoder":
        ckpt_dir = os.path.join(save_dir, "decoder")
        latest_dec = get_latest_checkpoint(ckpt_dir, "decoder_step_*.pth")
        
        if latest_dec:
            print(f"Found latest Decoder checkpoint: {latest_dec}")
            import re
            step = int(re.findall(r'\d+', os.path.basename(latest_dec))[0])
            
            # Find matching discriminator if it exists
            latest_disc = latest_dec.replace("decoder_step_", "discriminator_step_")
            disc_path = latest_disc if os.path.exists(latest_disc) else None

            update_config("config.yaml", {
                "paths": {
                    "pretrained_decoder_path": latest_dec,
                    "pretrained_discriminator_path": disc_path
                },
                "decoder": {"start_step": step}
            })
            print(f"Config updated to step {step}. Resuming Decoder training...")
            subprocess.run(["python", "train_decoder.py"])
        else:
            print("No Decoder checkpoints found to resume from.")

if __name__ == "__main__":
    main()