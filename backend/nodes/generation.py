import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from utils.memory import retrieve_knowledge, get_upcoming_events
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

async def generation_node(state):
    """
    Generates the final response using Mistral via NVIDIA.
    Executes after reasoning in sequential flow.
    """
    input_text = state["input_text"]
    username = state["username"]
    chat_history = state.get("chat_history", [])
    reasoning_context = state.get("reasoning_context", "")
    
    # 1. Retrieve Fast Context
    knowledge = await retrieve_knowledge(username, input_text)
    events = await get_upcoming_events(username)
    
    knowledge_str = "\n".join([f"- {k['fact']}" for k in knowledge]) if knowledge else "None"
    events_str = "\n".join([f"- {e['description']} at {e['event_time']}" for e in events]) if events else "None"
    
    user_profile = state.get("user_profile", {})
    name = user_profile.get("name", "User")
    
    # Properly format chat history
    chat_history_str = "\n".join(chat_history[-5:]) if chat_history else "No previous messages"

    # 2. Construct Prompt with Reasoning Context
    reasoning_section = f"""
    Recent Context (from reasoning):
    {reasoning_context}
    """ if reasoning_context else ""
    
    system_prompt = f"""You are an AI companion. You are talking to {name}.
    
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
    If the recent context shows a fact was just stored or an event was scheduled, acknowledge it warmly."""

    # 3. Generate Content with Mistral
    full_prompt = f"{system_prompt}\n\nUser: {input_text}"

    try:
        response = await client.ainvoke([HumanMessage(content=full_prompt)])
        final_text = response.content.strip()
        return {"final_response": final_text}

    except Exception as e:
        logger.error(f"Generation Error: {e}")
        return {"final_response": "I'm having trouble generating a response right now."}
