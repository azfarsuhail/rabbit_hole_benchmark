# Veritas: Objective-Anchoring in Agentic RAG

### Mitigating the "Rabbit Hole" Effect in Multi-Agent Tool Usage

This repository contains the benchmarking suite and core implementation for **Veritas**, an Agentic RAG system built to tackle LLM hallucinations and "distracted tool usage"—often referred to as the Rabbit Hole effect. By implementing **Objective-Anchoring** within LangGraph, Veritas keeps agents on track even in noisy semantic environments featuring over 100 available tools.

## 🚀 Overview

Standard RAG agents tend to get sidetracked when tool-retrieval layers (like ChromaDB) pull in irrelevant but semantically similar "bait" tools. Veritas addresses this through two main pillars:

1. **Semantic Tool Routing:** It uses **ChromaDB** paired with **HuggingFace Embeddings** to filter a library of 100+ tools down to the three most likely candidates.
2. **Objective-Anchoring:** This specialized LangGraph state-management technique requires the LLM to cross-reference every proposed tool call against the user's original intent before execution.

## 📊 Key Results

Testing across 500 adversarial queries revealed some pretty significant improvements:

* **10.2% Reduction in Token Tax:** A clear drop in unnecessary cumulative tool calls.
* **40% Improvement in Reliability:** We saw a major decrease in "long-tail" distraction events where an agent would previously spiral into 3 or more redundant steps.
* **Optimized Latency:** By cutting out useless execution cycles, average response times stayed much lower.

## 🛠 Tech Stack

* **Orchestration:** [LangGraph](https://github.com/langchain-ai/langgraph) for stateful multi-agent workflows.
* **LLM:** [Gemini 2.5 Flash](https://ai.google.dev/) via OpenRouter.
* **Vector Database:** [ChromaDB](https://www.trychroma.com/) running in a local Docker instance.
* **Embeddings:** `all-MiniLM-L6-v2` (HuggingFace).
* **Environment:** Docker & WSL2.

## 📁 Project Structure

```text
.
├── core/
│   ├── agent_setup.py      # LangGraph logic and Objective-Anchoring
│   └── retriever.py        # ChromaDB & HuggingFace embedding setup
├── data/
│   ├── baited_queries.csv  # The 500-query adversarial dataset
│   └── plots/              # Generated research visualizations
├── tools/
│   ├── core_tools.py       # Actual math and document utilities
│   ├── bait_tools.py       # Decoys designed to distract the model
│   └── dummy_tools.py      # Procedural generator for 100+ enterprise tools
├── run_benchmark.py        # The main execution pipeline
└── research_visualizer.py  # Script for 8 research-grade plots

```

## ⚡ Quick Start

### 1. Set up the environment

Drop your key into a `.env` file in the root:

```text
OPENROUTER_API_KEY=your_key_here

```

### 2. Boot the infrastructure

```bash
docker compose up -d

```

### 3. Run the benchmark

```bash
docker exec -it benchmark_app bash
python run_benchmark.py

```

### 4. Generate the visuals

```bash
python research_visualizer.py

```

## 📖 Citation

If you're using this benchmark or the Veritas architecture in your own work, please use the following citation:

```text
@article{veritas2026,
  title={Veritas: Objective-Anchoring for High-Precision Agentic RAG},
  author={Azfar Suhail},
  year={2026},
  publisher={GitHub}
}

```