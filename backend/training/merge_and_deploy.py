"""Merge LoRA adapter weights into base model for production inference.

After DPO training, run this to create a single merged model checkpoint
that can be loaded directly by local_mistral.py without adapter overhead.

Usage:
    python -m training.merge_and_deploy --config training/config/dpo_config.yaml
"""
import argparse
import yaml
import torch
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_config(config_path)

    base_model_path = cfg["base_model"]
    adapter_path = cfg.get("adapter_output", "./checkpoints/dpo-adapter")
    merged_output = cfg.get("merged_output", "./models/mistral-7b-improved")

    logger.info(f"Base model: {base_model_path}")
    logger.info(f"LoRA adapter: {adapter_path}")
    logger.info(f"Merged output: {merged_output}")

    # 1. Load base model in full precision for merging
    logger.info("Loading base model (full precision)...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16,
        device_map="cpu",  # Merge on CPU to avoid VRAM issues
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    # 2. Load LoRA adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # 3. Merge weights
    logger.info("Merging LoRA weights into base model...")
    model = model.merge_and_unload()

    # 4. Save merged model
    logger.info(f"Saving merged model to {merged_output}...")
    model.save_pretrained(merged_output, safe_serialization=True)
    tokenizer.save_pretrained(merged_output)

    logger.info(f"✓ Merged model saved to {merged_output}")
    logger.info(f"  Update LOCAL_MODEL_PATH in .env to: {merged_output}")
    logger.info(f"  Then restart the backend server.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--config", default="training/config/dpo_config.yaml")
    args = parser.parse_args()

    main(args.config)
