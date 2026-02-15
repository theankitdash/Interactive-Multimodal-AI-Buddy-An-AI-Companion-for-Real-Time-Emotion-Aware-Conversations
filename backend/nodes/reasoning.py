import logging
import json
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from utils.memory import store_knowledge, store_event
from datetime import datetime, timedelta, timezone
from config import NVIDIA_API_KEY, NVIDIA_MODEL, NVIDIA_TEMPERATURE, NVIDIA_TOP_P, NVIDIA_MAX_TOKENS

logger = logging.getLogger(__name__)

# Initialize Mistral Client via NVIDIA AI Endpoints
client = ChatNVIDIA(
    model=NVIDIA_MODEL,
    api_key=NVIDIA_API_KEY,
    temperature=NVIDIA_TEMPERATURE,
    top_p=NVIDIA_TOP_P,
    max_tokens=NVIDIA_MAX_TOKENS,
)

async def reasoning_node(state):
    """
    Analyzes input text for facts or events using Mistral.
    Executes before generation in sequential flow.
    """
    input_text = state["input_text"]
    username = state["username"]
    
    # 1. Classify Intent & Extract Info (Single Step for Speed)
    prompt = f"""Analyze the following user input: "{input_text}"
    
    Task:
    1. Classify intent into ONE category: FACT, EVENT, CHAT.
    2. IF FACT: Extract the core fact (e.g., "I like pizza" -> "User likes pizza").
    3. IF EVENT: Extract description and estimated time offset in minutes from now.
    
    Return JSON ONLY, no extra text:
    {{
        "category": "FACT" | "EVENT" | "CHAT",
        "fact": "extracted fact string" (if FACT),
        "event_description": "short description" (if EVENT),
        "time_offset_minutes": int (if EVENT, default 60 if unspecified)
    }}"""
    
    reasoning_context = ""
    
    try:
        response = await client.ainvoke([HumanMessage(content=prompt)])
        
        # Parse JSON from response
        result_text = response.content.strip()
        # Handle potential markdown code block wrapping
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
            result_text = result_text.strip()
        
        data = json.loads(result_text)
        
        category = data.get("category", "CHAT").upper()
        reasoning_context = f"Intent detected: {category}"
        
        if category == "FACT":
            fact = data.get("fact")
            if fact:
                await store_knowledge(username, fact, category="preference")
                reasoning_context += f". Stored fact: {fact}"
                logger.info(f"[Reasoning] ✓ Fact stored: {fact}")
                
        elif category == "EVENT":
            description = data.get("event_description", "Untitled Event")
            minutes = data.get("time_offset_minutes", 60)
            event_time = datetime.now(timezone.utc) + timedelta(minutes=minutes)
            
            try:
                await store_event(username, description, event_time)
                reasoning_context += f". Scheduled: {description} at {event_time.strftime('%I:%M %p')}"
                logger.info(f"[Reasoning] ✓ Event stored: {description} at {event_time}")
            except Exception as e:
                logger.error(f"[Reasoning] Failed to store event: {e}")
                reasoning_context += ". Failed to save event."

    except json.JSONDecodeError as e:
        logger.error(f"[Reasoning] JSON parse error: {e}")
        reasoning_context += ". Reasoning failed (invalid JSON)."
    except Exception as e:
        logger.error(f"[Reasoning] Mistral Error: {e}")
        reasoning_context += ". Reasoning failed."

    # Return partial state update
    return {"reasoning_context": reasoning_context}
