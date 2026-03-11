import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv

# Load your API key from the .env file
load_dotenv()

# We use Gemini 1.5 Flash as our Tier 2 model for fast, cheap, and reliable tool calling
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0, # Set to 0 so the agent is deterministic and doesn't hallucinate randomly
        max_output_tokens=500,
    )

def create_benchmark_agent(tools, use_anchoring=False):
    """
    Creates an agent. If use_anchoring is True, it applies your novel research fix 
    to prevent the agent from getting distracted by bait tools.
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
        # THE BASELINE: A standard, naive agent prompt (prone to getting distracted)
        system_instructions = (
            "You are a helpful AI assistant. Answer the user's questions and use tools "
            "if you think they would be helpful or interesting to the user."
        )

    # Build the prompt template required by LangChain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instructions),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # Construct the tool-calling agent
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # The AgentExecutor runs the loop (LLM -> Tool -> LLM -> Final Answer)
    # return_intermediate_steps=True is CRITICAL so we can count how many tools it wasted time on
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, 
        return_intermediate_steps=True 
    )
    
    return agent_executor