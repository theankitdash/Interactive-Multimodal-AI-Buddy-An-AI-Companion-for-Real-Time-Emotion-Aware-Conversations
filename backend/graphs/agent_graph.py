from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from nodes.reasoning import reasoning_node

# Define State
class AgentState(TypedDict):
    input_text: str
    username: str
    chat_history: List[str]
    user_profile: Dict[str, Any]
    vision_context: str  
    reasoning_context: str
    final_response: str
    audio_mode: bool  # If True, skip generation (Gemini Live handles response)

# Define Graph
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("reasoning", reasoning_node)

# Conditional: Skip generation in audio mode
def should_generate(state) -> str:
    """Route to generation or skip based on audio mode."""
    if state.get("audio_mode", False):
        return "end"  # Skip generation, Gemini Live Audio handles it
    return "generate"

# Import generation only when needed (to avoid circular import issues)
async def generation_wrapper(state):
    """Wrapper that only generates if not in audio mode."""
    if state.get("audio_mode", False):
        # In audio mode, just return empty response (Gemini Live handles speech)
        return {"final_response": ""}
    
    # Import here to avoid issues
    from nodes.generation import generation_node
    return await generation_node(state)

workflow.add_node("generation", generation_wrapper)

# Sequential execution: Reasoning first, then conditional generation
# Note: LangGraph executes nodes sequentially along edges, not in parallel
workflow.set_entry_point("reasoning")

# After reasoning, check if we should generate
workflow.add_conditional_edges(
    "reasoning",
    should_generate,
    {
        "generate": "generation",
        "end": END
    }
)

workflow.add_edge("generation", END)

# Compile the graph
app = workflow.compile()
