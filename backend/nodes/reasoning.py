import logging
import json
from langchain_core.messages import HumanMessage
from ai.nvidia_client import mistral_client as client
from utils.memory import store_knowledge, store_event
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

async def reasoning_node(state):
    input_text = state["input_text"]
    username = state["username"]
    chat_history = state.get("chat_history", [])
    user_profile = state.get("user_profile", {})
    vision_context = state.get("vision_context", "")
    
    # Build context sections
    name = user_profile.get("name", username)
    
    history_str = ""
    if chat_history:
        recent = chat_history[-8:]  # Last 8 turns for context
        history_str = f"\n\nRecent conversation:\n" + "\n".join(recent)
    
    vision_section = ""
    if vision_context and vision_context != "Camera is off. No visual data available.":
        vision_section = f"\nVisual context: {vision_context}"
    
    prompt = f"""You are analyzing a live conversation with {name}.{history_str}{vision_section}

Latest message from {name}: "{input_text}"

Your job: classify this message and extract structured data.

RULES:
- CHAT → The user is asking a question, making a request, greeting, or just chatting. This is the DEFAULT. When in doubt, use CHAT.
- FACT → The user explicitly shares personal information about themselves. Must be a declarative statement, NOT a question.
  Subtypes: "preference" (likes, dislikes, favorites, personal details like age/job) or "memory" (past experiences, stories, things that happened to them).
  Examples: "I love hiking" → preference. "I visited Japan last year" → memory. "My name is Alex" → preference.
- EVENT → The user wants to SCHEDULE something in the FUTURE. Must include a time reference.
  Examples: "Remind me to call mom in 30 minutes". "I have a dentist appointment tomorrow at 4pm".
  Sharing past events is NOT this category — that's FACT/memory.

Return ONLY valid JSON:
{{
    "category": "CHAT" | "FACT" | "EVENT",
    "fact": "concise extracted fact" (only if FACT),
    "fact_type": "preference" | "memory" (only if FACT),
    "event_description": "short description" (only if EVENT),
    "time_offset_minutes": number (only if EVENT, default 60)
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
                fact_type = data.get("fact_type", "other")
                # Map fact_type to knowledge_category enum
                cat = fact_type if fact_type in ("preference", "memory") else "other"
                await store_knowledge(username, fact, category=cat)
                reasoning_context += f". Stored {cat}: {fact}"
                logger.info(f"[Reasoning] ✓ {cat.title()} stored: {fact}")
                
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

