"""Evaluate model quality before and after DPO training.

Runs a standard test suite comparing the base model vs fine-tuned model
on intent classification accuracy and response quality.

Usage:
    python -m training.evaluate --model-path mistralai/Mistral-7B-Instruct-v0.3
    python -m training.evaluate --model-path ./models/mistral-7b-improved
"""
import argparse
import json
import torch
import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard test cases for intent classification (reasoning node)
INTENT_TEST_CASES = [
    {"input": "Hello, how are you?", "expected_category": "CHAT"},
    {"input": "I love hiking in the mountains", "expected_category": "FACT"},
    {"input": "My favorite color is blue", "expected_category": "FACT"},
    {"input": "Remind me to call mom in 30 minutes", "expected_category": "EVENT"},
    {"input": "What's the weather like today?", "expected_category": "CHAT"},
    {"input": "I visited Japan last summer", "expected_category": "FACT"},
    {"input": "Set an alarm for 7 AM tomorrow", "expected_category": "EVENT"},
    {"input": "Tell me a joke", "expected_category": "CHAT"},
    {"input": "I'm allergic to peanuts", "expected_category": "FACT"},
    {"input": "Schedule a meeting at 3 PM today", "expected_category": "EVENT"},
]


def build_reasoning_prompt(user_input: str) -> str:
    """Build the same reasoning prompt used in reasoning.py."""
    return f"""You are analyzing a live conversation with User.

Latest message from User: "{user_input}"

Your job: classify this message and extract structured data.

RULES:
- CHAT → The user is asking a question, making a request, greeting, or just chatting. This is the DEFAULT.
- FACT → The user explicitly shares personal information about themselves.
  Subtypes: "preference" or "memory".
- EVENT → The user wants to SCHEDULE something in the FUTURE. Must include a time reference.

Return ONLY valid JSON:
{{
    "category": "CHAT" | "FACT" | "EVENT",
    "fact": "concise extracted fact" (only if FACT),
    "fact_type": "preference" | "memory" (only if FACT),
    "event_description": "short description" (only if EVENT),
    "time_offset_minutes": number (only if EVENT, default 60)
}}"""


def main(model_path: str):
    logger.info(f"Evaluating model: {model_path}")

    # Load model with 4-bit quantization
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        ),
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.float16,
    )
    model.eval()

    # Run intent classification tests
    json_valid = 0
    category_correct = 0
    total = len(INTENT_TEST_CASES)

    for tc in INTENT_TEST_CASES:
        prompt = build_reasoning_prompt(tc["input"])

        try:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            formatted = f"<s>[INST] {prompt} [/INST]"

        inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=256, temperature=0.1,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Check JSON validity
        try:
            # Handle markdown code blocks
            text = response
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            data = json.loads(text)
            json_valid += 1

            # Check category
            if data.get("category", "").upper() == tc["expected_category"]:
                category_correct += 1
                status = "✓"
            else:
                status = f"✗ (got {data.get('category', 'N/A')})"
        except json.JSONDecodeError:
            status = "✗ (invalid JSON)"

        logger.info(f"  [{status}] \"{tc['input']}\" → expected {tc['expected_category']}")

    # Summary
    print("\n" + "=" * 50)
    print(f"Model: {model_path}")
    print(f"JSON valid:       {json_valid}/{total} ({100*json_valid/total:.0f}%)")
    print(f"Category correct: {category_correct}/{total} ({100*category_correct/total:.0f}%)")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on standard test suite")
    parser.add_argument("--model-path", default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    main(args.model_path)
