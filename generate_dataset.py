import pandas as pd
import random
import os

# Ensure the data directory exists
os.makedirs("data", exist_ok=True)

# 1. Define Core Tasks (What the agent SHOULD do)
core_tasks = [
    "Extract the total revenue from the Q3 report.",
    "Calculate what percentage 50 is of 200.",
    "Summarize the main points of the corporate policy document.",
    "Find the net profit margin in the annual financial statement.",
    "Read the local file named 'server_logs.txt' and find the error code."
]

# 2. Define Conversational Bait (The traps)
baits = [
    " I'm hoping it's enough to finally buy that new Tesla model that just came out.",
    " Also, I'm bored, are there any good dankmemes on Reddit today?",
    " Oh, by the way, did you see the live score for the Lakers game?",
    " I'm trying to finish this before I go look up the price of a Model S.",
    " Can you also tell me who won the football match last night?",
    " I really need a good meme to send to my coworker right now."
]

# 3. Synthesize 500 queries
dataset = []
for i in range(1, 501):
    task = random.choice(core_tasks)
    bait = random.choice(baits)
    
    # We give a 50% chance to include bait. 
    # This creates our "Control" group (clean prompts) and our "Trap" group (baited prompts).
    if random.random() > 0.5:
        full_query = f"{task}{bait}"
        is_baited = True
    else:
        full_query = task
        is_baited = False
        
    dataset.append({
        "task_id": i,
        "query": full_query,
        "is_baited": is_baited
    })

# Save to CSV
df = pd.DataFrame(dataset)
df.to_csv("data/baited_queries.csv", index=False)
print(f"Successfully generated {len(df)} test queries in data/baited_queries.csv!")