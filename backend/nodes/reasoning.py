import google.genai as genai
from google.genai import types
from utils.memory import store_knowledge, store_event
from datetime import datetime, timedelta
import json
import os
import logging
from config import GEMINI_MODEL

logger = logging.getLogger(__name__)

# Initialize Gemini Client for Reasoning
# Using the same client/model as generation to unify the stack
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def reasoning_node(state):
    """
    Analyzes input text for facts or events using Gemini.
    Executes before generation in sequential flow.
    """
    input_text = state["input_text"]
    username = state["username"]
    
    # 1. Classify Intent & Extract Info (Single Step for Speed)
    # Gemini Flash is fast enough to do classification and extraction in one prompt
    prompt = f"""
    Analyze the following user input: "{input_text}"
    
    Task:
    1. Classify intent into ONE category: FACT, EVENT, CHAT.
    2. IF FACT: Extract the core fact (e.g., "I like pizza" -> "User likes pizza").
    3. IF EVENT: Extract description and estimated time offset in minutes from now.
    
    Return JSON ONLY:
    {{
        "category": "FACT" | "EVENT" | "CHAT",
        "fact": "extracted fact string" (if FACT),
        "event_description": "short description" (if EVENT),
        "time_offset_minutes": int (if EVENT, default 60 if unspecified)
    }}
    """
    
    reasoning_context = ""
    
    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"],
                response_mime_type="application/json"  # Enforce JSON output
            )
        )
        
        # Parse output
        result_text = response.candidates[0].content.parts[0].text
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
            event_time = datetime.now() + timedelta(minutes=minutes)
            
            try:
                await store_event(username, description, event_time)
                reasoning_context += f". Scheduled: {description} at {event_time.strftime('%I:%M %p')}"
                logger.info(f"[Reasoning] ✓ Event stored: {description} at {event_time}")
            except Exception as e:
                logger.error(f"[Reasoning] Failed to store event: {e}")
                reasoning_context += ". Failed to save event."

    except Exception as e:
        logger.error(f"[Reasoning] Gemini Error: {e}")
        reasoning_context += ". Reasoning failed."

    # Return partial state update
    return {"reasoning_context": reasoning_context}
