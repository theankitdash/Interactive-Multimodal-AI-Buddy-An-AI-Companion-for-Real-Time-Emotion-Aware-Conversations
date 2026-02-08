from langchain_nvidia_ai_endpoints import ChatNVIDIA
from utils.memory import store_knowledge, store_event
from datetime import datetime, timedelta
import json
import os

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
        # Simple extraction for demo - in production, use structured output or specific prompts for time
        extract_prompt = f"""
        Extract event details from: '{input_text}'.
        Return JSON with keys: description, time_offset_minutes (int, estimate).
        Example: {{"description": "Buy milk", "time_offset_minutes": 60}}
        """
        try:
            event_resp = await llm.ainvoke(extract_prompt)
            # Basic parsing logic (Mistral might output text around JSON)
            content = event_resp.content.strip()
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end != -1:
                event_data = json.loads(content[start:end])
                description = event_data.get("description", "Untitled Event")
                minutes = event_data.get("time_offset_minutes", 60)
                event_time = datetime.now() + timedelta(minutes=minutes)
                
                await store_event(username, description, event_time)
                reasoning_context += f". Scheduled: {description} at {event_time}"
        except Exception as e:
            reasoning_context += f". Failed to extract event: {e}"

    # Return partial state update
    return {"reasoning_context": reasoning_context}
