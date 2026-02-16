"""Shared NVIDIA Mistral client for reasoning and generation nodes."""
import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from config import NVIDIA_API_KEY, NVIDIA_MODEL, NVIDIA_TEMPERATURE, NVIDIA_TOP_P, NVIDIA_MAX_TOKENS

logger = logging.getLogger(__name__)

# Single shared Mistral client via NVIDIA AI Endpoints
mistral_client = ChatNVIDIA(
    model=NVIDIA_MODEL,
    api_key=NVIDIA_API_KEY,
    temperature=NVIDIA_TEMPERATURE,
    top_p=NVIDIA_TOP_P,
    max_tokens=NVIDIA_MAX_TOKENS,
)
