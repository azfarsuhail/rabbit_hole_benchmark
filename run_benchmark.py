import time
import pandas as pd
import warnings
from tools.core_tools import core_tools_list
from tools.bait_tools import bait_tools_list
from tools.dummy_tools import generate_dummy_tools
from core.agent_setup import create_benchmark_agent
from core.retriever import setup_tool_retriever

warnings.filterwarnings("ignore")

# 1. Assemble the massive 100+ tool library
print("Generating 100 dummy tools...")
dummy_tools_list = generate_dummy_tools(100)
all_tools = core_tools_list + bait_tools_list + dummy_tools_list

def run_evaluation():
    print("Loading dataset...")
    try:
        df = pd.read_csv("data/baited_queries.csv")
        # df = df.head(5) # Uncomment to run a quick test
    except FileNotFoundError:
        print("Error: data/baited_queries.csv not found.")
        return

    # 2. Boot up ChromaDB and embed all 100+ tools (This takes ~10 seconds)
    print(f"Embedding {len(all_tools)} tools into ChromaDB vector space...")
    retriever = setup_tool_retriever(all_tools)

    results = []
    print("\nStarting the Agentic RAG Benchmark...\n" + "-"*40)

    for index, row in df.iterrows():
        query = row["query"]
        print(f"Testing Query {row['task_id']}...")

        # --- THE MAGIC HAPPENS HERE: SEMANTIC ROUTING ---
        # Ask Chroma for the Top 3 most relevant tools based on the user's query
        retrieved_docs = retriever.invoke(query)
        retrieved_tool_names = [doc.metadata["name"] for doc in retrieved_docs]
        
        # Filter our massive 100+ tool library down to JUST those 3 tools
        active_tools = [t for t in all_tools if t.name in retrieved_tool_names]
        
        print(f" -> Chroma retrieved: {retrieved_tool_names}")

        # 3. Build the agents dynamically using ONLY the retrieved tools
        baseline_agent = create_benchmark_agent(active_tools, use_anchoring=False)
        anchored_agent = create_benchmark_agent(active_tools, use_anchoring=True)

        # --- TEST 1: BASELINE AGENT ---
        start_time = time.time()
        baseline_response = baseline_agent.invoke({"messages": [("user", query)]})
        baseline_latency = time.time() - start_time
        
        baseline_steps = len([m for m in baseline_response["messages"] if m.type == "tool"])
        baseline_final_answer = baseline_response["messages"][-1].content

        # --- TEST 2: ANCHORED AGENT ---
        start_time = time.time()
        anchored_response = anchored_agent.invoke({"messages": [("user", query)]})
        anchored_latency = time.time() - start_time
        
        anchored_steps = len([m for m in anchored_response["messages"] if m.type == "tool"])
        anchored_final_answer = anchored_response["messages"][-1].content

        results.append({
            "task_id": row["task_id"],
            "query": query,
            "baseline_steps_taken": baseline_steps,
            "baseline_latency_sec": round(baseline_latency, 2),
            "anchored_steps_taken": anchored_steps,
            "anchored_latency_sec": round(anchored_latency, 2),
            "baseline_final_answer": baseline_final_answer,
            "anchored_final_answer": anchored_final_answer
        })

        print(f" -> Baseline used {baseline_steps} tools.")
        print(f" -> Anchored used {anchored_steps} tools.")
        print("-" * 40)

    results_df = pd.DataFrame(results)
    results_df.to_csv("data/benchmark_results.csv", index=False)
    print("\nBenchmark complete! Empirical data saved to data/benchmark_results.csv")

if __name__ == "__main__":
    run_evaluation()