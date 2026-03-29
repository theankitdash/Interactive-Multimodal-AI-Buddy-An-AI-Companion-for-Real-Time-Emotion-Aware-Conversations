"""Local Mistral 7B client — drop-in replacement for ChatNVIDIA.

Loads Mistral-7B-Instruct-v0.3 from HuggingFace with 4-bit quantization
and wraps it as a LangChain-compatible chat model so that reasoning.py
and generation.py need zero code changes.
"""
import asyncio
import logging
import functools
from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from pydantic import Field

import config

logger = logging.getLogger(__name__)


def _format_messages_to_mistral(messages: List[BaseMessage], tokenizer) -> str:
    """Convert LangChain messages into Mistral instruct format.

    Mistral instruct format: <s>[INST] {user_msg} [/INST] {assistant_response}</s>
    We combine system + human messages into a single [INST] block since Mistral
    doesn't have a native system role — it uses the first [INST] block.
    """
    parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(msg.content)
        elif isinstance(msg, HumanMessage):
            parts.append(msg.content)

    combined = "\n\n".join(parts)

    # Use the tokenizer's chat template if available, otherwise manual format
    try:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": combined}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return formatted
    except Exception:
        # Fallback to manual Mistral instruct format
        return f"<s>[INST] {combined} [/INST]"


class LocalMistralClient(BaseChatModel):
    """LangChain-compatible wrapper for locally-loaded Mistral 7B.

    Uses bitsandbytes 4-bit quantization for ~6 GB VRAM usage.
    Runs model.generate() in a thread pool executor for async compatibility.
    """

    model_path: str = ""
    temperature: float = 0.2
    top_p: float = 0.7
    max_tokens: int = 1024
    quantize_4bit: bool = True

    # Internal state (not serialized by pydantic)
    _model: Any = None
    _tokenizer: Any = None
    _device: str = "cuda"

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, model_path: str, temperature: float = 0.2,
                 top_p: float = 0.7, max_tokens: int = 1024,
                 quantize_4bit: bool = True, **kwargs):
        super().__init__(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            quantize_4bit=quantize_4bit,
            **kwargs,
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load model and tokenizer from HuggingFace / local path."""
        logger.info(f"Loading Mistral model from: {self.model_path}")
        logger.info(f"Device: {self._device} | 4-bit quantization: {self.quantize_4bit}")

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Quantization config
        model_kwargs = {"device_map": "auto", "dtype": torch.float16}
        if self.quantize_4bit and self._device == "cuda":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # Model
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        self._model.eval()
        logger.info("✓ Mistral model loaded successfully")

    @property
    def _llm_type(self) -> str:
        return "local-mistral-7b"

    def _generate(self, messages: List[List[BaseMessage]], stop: Optional[List[str]] = None,
                  run_manager=None, **kwargs) -> List[ChatResult]:
        """Synchronous generation — required by BaseChatModel."""
        results = []
        for msg_list in messages:
            text = self._run_inference(msg_list)
            results.append(ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            ))
        return results

    def _run_inference(self, messages: List[BaseMessage]) -> str:
        """Run model.generate() synchronously."""
        prompt = _format_messages_to_mistral(messages, self._tokenizer)

        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=max(self.temperature, 0.01),  # Avoid 0.0
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # Decode only the NEW tokens (skip the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response

    async def ainvoke(self, input: Any, config: Any = None, **kwargs) -> AIMessage:
        """Async inference — runs generation in a thread pool to avoid blocking."""
        messages = input if isinstance(input, list) else [input]

        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(
            None,
            functools.partial(self._run_inference, messages),
        )
        return AIMessage(content=text)

    async def _agenerate(self, messages: List[List[BaseMessage]], stop=None,
                         run_manager=None, **kwargs):
        """Async generation — required by BaseChatModel."""
        results = []
        for msg_list in messages:
            loop = asyncio.get_running_loop()
            text = await loop.run_in_executor(
                None,
                functools.partial(self._run_inference, msg_list),
            )
            results.append(ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            ))
        return results


# ─── Singleton instance ───────────────────────────────────────────────
# Loaded once when the module is first imported (server startup).
# Uses the path from config — defaults to HuggingFace hub ID for first run,
# later points to a locally-merged fine-tuned checkpoint.

logger.info("Initializing local Mistral client...")
mistral_client = LocalMistralClient(
    model_path=config.LOCAL_MODEL_PATH,
    temperature=config.MISTRAL_TEMPERATURE,
    top_p=config.MISTRAL_TOP_P,
    max_tokens=config.MISTRAL_MAX_TOKENS,
    quantize_4bit=config.MISTRAL_QUANTIZE_4BIT,
)
