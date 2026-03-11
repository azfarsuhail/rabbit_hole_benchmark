import time
import pandas as pd
import warnings
from tools.core_tools import core_tools_list
from tools.bait_tools import bait_tools_list
from core.agent_setup import create_benchmark_agent

# Suppress minor LangChain deprecation warnings for clean terminal output
warnings.filterwarnings("ignore")

# 1. Combine all tools for the environment
all_tools = core_tools_list + bait_tools_list

# 2. Mock Dataset (For testing the pipeline before scaling up)
# Notice how each query contains a Core Task AND conversational "Bait"
# 2. Load the Full Dataset
print("Loading dataset...")
try:
    df = pd.read_csv("data/baited_queries.csv")
    # For testing, you might want to slice it: df = df.head(10)
    # Remove .head() to run all 500 when you are ready for the final benchmark
except FileNotFoundError:
    print("Error: data/baited_queries.csv not found. Run generate_dataset.py first.")
    exit()

def run_evaluation():
    print("Initializing Agents...")
    # The industry standard (prone to distractions)
    baseline_agent = create_benchmark_agent(all_tools, use_anchoring=False)
    # Your proposed research fix
    anchored_agent = create_benchmark_agent(all_tools, use_anchoring=True)

    results = []

    print("\nStarting the Rabbit Hole Benchmark...\n" + "-"*40)

    # 3. The Benchmarking Loop
    for index, row in df.iterrows():
        query = row["query"]
        print(f"Testing Query {row['task_id']}...")

        # --- TEST 1: BASELINE AGENT ---
        start_time = time.time()
        # .invoke() runs the entire LangChain reasoning loop
        baseline_response = baseline_agent.invoke({"input": query})
        baseline_latency = time.time() - start_time
        # Count how many tools it decided to use
        baseline_steps = len(baseline_response.get("intermediate_steps", []))

        # --- TEST 2: ANCHORED AGENT (Your Fix) ---
        start_time = time.time()
        anchored_response = anchored_agent.invoke({"input": query})
        anchored_latency = time.time() - start_time
        anchored_steps = len(anchored_response.get("intermediate_steps", []))

        # 4. Log the empirical metrics
        results.append({
            "task_id": row["task_id"],
            "query": query,
            "baseline_steps_taken": baseline_steps,
            "baseline_latency_sec": round(baseline_latency, 2),
            "anchored_steps_taken": anchored_steps,
            "anchored_latency_sec": round(anchored_latency, 2),
            "baseline_final_answer": baseline_response["output"],
            "anchored_final_answer": anchored_response["output"]
        })

        print(f" -> Baseline used {baseline_steps} tools.")
        print(f" -> Anchored used {anchored_steps} tools.")
        print("-" * 40)

    # 5. Save the final data to a CSV for your research paper
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/benchmark_results.csv", index=False)
    print("\nBenchmark complete! Empirical data saved to data/benchmark_results.csv")

if __name__ == "__main__":
    run_evaluation()