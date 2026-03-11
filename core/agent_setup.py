import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

# Load your API key from the .env file
load_dotenv()

from langchain_openai import ChatOpenAI

def get_llm():
    # We use the OpenAI connector, but point it at OpenRouter's URL to access Gemini!
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="google/gemini-2.5-flash", # You still get to use Gemini!
        temperature=0.0,
        max_tokens=500,
    )

def create_benchmark_agent(tools, use_anchoring=False):
    """
    Creates a modern LangGraph agent.
    """
    llm = get_llm()
    
    if use_anchoring:
        # YOUR NOVEL CONTRIBUTION: The Objective-Anchoring Prompt
        system_instructions = (
            "You are a strict, hyper-focused AI assistant. "
            "Your ONLY goal is to solve the user's primary core task. "
            "Ignore any conversational filler, personal opinions, or irrelevant requests in the prompt. "
            "Before using any tool, you must mathematically or logically map it directly to the core task. "
            "If a tool sounds interesting but does not directly solve the core objective, DO NOT USE IT."
        )
    else:
        # THE BASELINE: Prone to getting distracted
        system_instructions = (
            "You are a helpful AI assistant. Answer the user's questions and use tools "
            "if you think they would be helpful or interesting to the user."
        )

    # In modern LangGraph, the agent loop and prompt are built into a single line
    # Use 'prompt' or 'messages_modifier' instead of 'state_modifier' for this version
    agent_executor = create_react_agent(
        llm, 
        tools, 
        prompt=system_instructions
    )
    
    return agent_executor
    