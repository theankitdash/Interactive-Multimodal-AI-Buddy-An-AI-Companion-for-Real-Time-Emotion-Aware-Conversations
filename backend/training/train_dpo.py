"""DPO fine-tuning with LoRA for continuous model improvement.

Trains on preference pairs collected from real user interactions.
Uses Direct Preference Optimization — no separate reward model needed.

Usage:
    python -m training.train_dpo --config training/config/dpo_config.yaml

Requirements:
    - GPU with >= 16 GB VRAM (or use Google Colab with T4/A100)
    - DPO pairs exported via: python -m training.export_feedback
"""
import argparse
import os
import sys
import yaml
import torch
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str, data_path: str):
    cfg = load_config(config_path)

    logger.info(f"Base model: {cfg['base_model']}")
    logger.info(f"Data: {data_path}")

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Quantization config
    quant_cfg = cfg.get("quantization", {})
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.get("load_in_4bit", True),
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")),
        bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
    )

    # 3. Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    # 4. LoRA config
    lora_cfg = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        bias="none",
    )

    # 5. Load DPO preference pairs
    dataset = load_dataset("json", data_files=data_path, split="train")
    logger.info(f"Loaded {len(dataset)} preference pairs")

    # 6. DPO training config
    dpo_cfg = cfg.get("dpo", {})
    training_args = DPOConfig(
        output_dir=cfg.get("adapter_output", "./checkpoints/dpo-adapter"),
        beta=dpo_cfg.get("beta", 0.1),
        num_train_epochs=dpo_cfg.get("num_epochs", 1),
        per_device_train_batch_size=dpo_cfg.get("batch_size", 2),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=dpo_cfg.get("learning_rate", 5e-5),
        lr_scheduler_type=dpo_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=dpo_cfg.get("warmup_ratio", 0.03),
        max_length=dpo_cfg.get("max_seq_length", 1024),
        fp16=dpo_cfg.get("fp16", True),
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
    )

    # 7. Train
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPO infers reference from base model
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    logger.info("Starting DPO training...")
    trainer.train()

    # 8. Save adapter
    adapter_path = cfg.get("adapter_output", "./checkpoints/dpo-adapter")
    trainer.save_model(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"✓ LoRA adapter saved to {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO fine-tuning with LoRA")
    parser.add_argument("--config", default="training/config/dpo_config.yaml")
    parser.add_argument("--data", default="training/data/dpo_pairs.jsonl")
    args = parser.parse_args()

    main(args.config, args.data)
