from langchain_nvidia_ai_endpoints import ChatNVIDIA
from utils.memory import store_knowledge, store_event
from datetime import datetime, timedelta
import json
import os
import logging

logger = logging.getLogger(__name__)

# Initialize Mistral
# Using the same model config as assistant_orchestrator.py
llm = ChatNVIDIA(
    model="mistralai/mistral-7b-instruct-v0.3",
    api_key=os.getenv("NVIDIA_API_KEY"),
    temperature=0.1,
    max_completion_tokens=512
)

async def reasoning_node(state):
    """
    Analyzes input text for facts or events.
    Executes before generation in sequential flow.
    """
    input_text = state["input_text"]
    username = state["username"]
    
    # 1. Classify Intent
    prompt = f"""
    Analyze the following user input.
    Classify into ONE category: FACT, EVENT, CHAT.
    
    - FACT: User defines a preference, habit, or memory (e.g., "I like pizza", "My birthday is in June").
    - EVENT: User mentions a task, meeting, or reminder (e.g., "Remind me to buy milk", "Meeting tomorrow at 9").
    - CHAT: General conversation.

    Input: {input_text}
    Output (Category only):
    """
    resp = await llm.ainvoke(prompt)
    category = resp.content.strip().upper()
    
    reasoning_context = f"Intent detected: {category}"

    # 2. Extract & Store Information
    if "FACT" in category:
        extract_prompt = f"Extract the core fact from: '{input_text}'. Return ONLY the fact statement."
        fact_resp = await llm.ainvoke(extract_prompt)
        fact = fact_resp.content.strip()
        await store_knowledge(username, fact, category="preference")
        reasoning_context += f". Stored fact: {fact}"

    elif "EVENT" in category:
        # Extract event details using Mistral
        extract_prompt = f"""
        Extract event details from: '{input_text}'.
        Return JSON with keys: description, time_offset_minutes (int, estimate).
        Example: {{"description": "Buy milk", "time_offset_minutes": 60}}
        """
        
        # Step 1: Get LLM response
        try:
            event_resp = await llm.ainvoke(extract_prompt)
            content = event_resp.content.strip()
            logger.info(f"[Reasoning] Mistral response for event: {content}")
            
            # Step 2: Parse JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            
            if start == -1 or end == -1:
                logger.warning(f"[Reasoning] No JSON found in Mistral response: {content}")
                reasoning_context += ". Failed to extract event: No JSON in response"
            else:
                try:
                    event_data = json.loads(content[start:end])
                    description = event_data.get("description", "Untitled Event")
                    minutes = event_data.get("time_offset_minutes", 60)
                    event_time = datetime.now() + timedelta(minutes=minutes)
                    
                    logger.info(f"[Reasoning] Parsed event: {description} at {event_time} (in {minutes} min)")
                    
                    # Step 3: Store to database
                    try:
                        await store_event(username, description, event_time)
                        logger.info(f"[Reasoning] ✓ Event stored successfully for {username}")
                        reasoning_context += f". Scheduled: {description} at {event_time.strftime('%I:%M %p')}"
                    except Exception as db_error:
                        logger.error(f"[Reasoning] ✗ Database error storing event: {db_error}", exc_info=True)
                        reasoning_context += f". Failed to save event: Database error"
                        
                except json.JSONDecodeError as json_error:
                    logger.error(f"[Reasoning] ✗ JSON parsing error: {json_error}. Content: {content[start:end]}")
                    reasoning_context += f". Failed to extract event: Invalid JSON format"
                    
        except Exception as llm_error:
            logger.error(f"[Reasoning] ✗ LLM invocation error: {llm_error}", exc_info=True)
            reasoning_context += f". Failed to extract event: LLM error"

    # Return partial state update
    return {"reasoning_context": reasoning_context}
