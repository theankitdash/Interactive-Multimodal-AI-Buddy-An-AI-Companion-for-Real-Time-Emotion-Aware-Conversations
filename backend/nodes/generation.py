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
    Runs in parallel with reasoning (but can start immediately).
    """
    input_text = state["input_text"]
    username = state["username"]
    chat_history = state.get("chat_history", [])
    
    # 1. Retrieve Fast Context
    # (Reasoning node might be too slow, so we do a quick lookup here too if needed, 
    # or rely on what's passed in. For immediate response, we do a quick retrieval).
    knowledge = await retrieve_knowledge(username, input_text)
    events = await get_upcoming_events(username)
    
    knowledge_str = "\n".join([f"- {k['fact']}" for k in knowledge])
    events_str = "\n".join([f"- {e['description']} at {e['event_time']}" for e in events])
    
    user_profile = state.get("user_profile", {})
    name = user_profile.get("name", "User")

    # 2. Construct Prompt
    system_prompt = f"""
    You are an AI companion. You are talking to {name}.
    
    User Profile:
    - Name: {name}
    
    Relevant Memories:
    {knowledge_str}
    
    Upcoming Events:
    {events_str}
    
    Chat History:
    {chat_history[-5:]}
    
    Respond naturally, empathetically, and concisely to the user.
    """
    
    # 3. Generate Content
    # Using raw genai client for text generation
    # Explicitly request TEXT to avoid audio/multipart complexity in this node
    response = await client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            {"role": "user", "parts": [{"text": system_prompt + f"\nUser: {input_text}"}]}
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
