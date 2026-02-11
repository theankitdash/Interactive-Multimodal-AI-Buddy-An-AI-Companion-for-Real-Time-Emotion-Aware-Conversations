import os
import google.genai as genai
from utils.memory import retrieve_knowledge, get_upcoming_events
from config import GEMINI_MODEL

from google.genai import types

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

async def generation_node(state):
    """
    Generates the final response using Gemini.
    Executes after reasoning in sequential flow.
    """
    input_text = state["input_text"]
    username = state["username"]
    chat_history = state.get("chat_history", [])
    reasoning_context = state.get("reasoning_context", "")
    
    # 1. Retrieve Fast Context
    # (Reasoning node might be too slow, so we do a quick lookup here too if needed, 
    # or rely on what's passed in. For immediate response, we do a quick retrieval).
    knowledge = await retrieve_knowledge(username, input_text)
    events = await get_upcoming_events(username)
    
    knowledge_str = "\n".join([f"- {k['fact']}" for k in knowledge]) if knowledge else "None"
    events_str = "\n".join([f"- {e['description']} at {e['event_time']}" for e in events]) if events else "None"
    
    user_profile = state.get("user_profile", {})
    name = user_profile.get("name", "User")
    
    # Properly format chat history
    chat_history_str = "\n".join(chat_history[-5:]) if chat_history else "No previous messages"

    # 2. Construct Prompt with Reasoning Context
    # Include what the reasoning node discovered so we can immediately acknowledge facts/events
    reasoning_section = f"""
    Recent Context (from reasoning):
    {reasoning_context}
    """ if reasoning_context else ""
    
    system_prompt = f"""
    You are an AI companion. You are talking to {name}.
    
    User Profile:
    - Name: {name}
    
    Relevant Memories:
    {knowledge_str}
    
    Upcoming Events:
    {events_str}
    
    Chat History:
    {chat_history_str}
    {reasoning_section}
    Respond naturally, empathetically, and concisely to the user.
    If the recent context shows a fact was just stored or an event was scheduled, acknowledge it warmly.
    """
    
    # 3. Generate Content
    # Using raw genai client for text generation
    # Explicitly request TEXT to avoid audio/multipart complexity in this node
    
    # Prepare contents - if images were in state, we would add them here
    # For now, we just send text, but the model is ready for images
    prompt_parts = [{"text": system_prompt + f"\nUser: {input_text}"}]
    
    # Future Vision Integration:
    # if "image_data" in state:
    #     prompt_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": state["image_data"]}})

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {"role": "user", "parts": prompt_parts}
            ],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT"]
            )
        )
        
        # Manually extract text to avoid warnings about "thought" or "non-data" parts
        final_text = ""
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                # Check if likely a thought part (using safe getattr in case of SDK version diffs)
                is_thought = getattr(part, 'thought', False)
                if is_thought:
                    continue
                    
                # Only append if valid text exists content
                if part.text:
                    final_text += part.text
                    
        return {"final_response": final_text}

    except Exception as e:
        import logging
        logging.error(f"Generation Error: {e}")
        return {"final_response": "I'm having trouble generating a response right now."}
