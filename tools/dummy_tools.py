from langchain_core.tools import Tool

def generate_dummy_tools(num_tools=100):
    """Procedurally generates hundreds of decoy tools to test semantic retrieval."""
    dummy_tools = []
    
    for i in range(num_tools):
        # A simple dummy function that the tool will execute if chosen
        def dummy_func(query: str, idx=i) -> str:
            return f"Dummy database {idx} queried successfully."
        
        # We give them realistic, slightly confusing names and descriptions
        tool = Tool(
            name=f"enterprise_metric_fetcher_{i}",
            func=dummy_func,
            description=(
                f"Fetches enterprise metrics and telemetry from database shard {i}. "
                "Use this if the user asks about system logs, user telemetry, or generic corporate data."
            )
        )
        dummy_tools.append(tool)
        
    return dummy_tools