import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_visualizations():
    print("Loading benchmark data...")
    try:
        df = pd.read_csv("data/benchmark_results.csv")
    except FileNotFoundError:
        print("Error: Could not find data/benchmark_results.csv. Run the benchmark first!")
        return

    # Create a folder for the graphs
    os.makedirs("data/plots", exist_ok=True)
    
    # Set the visual style for research-paper quality graphs
    sns.set_theme(style="whitegrid")

    # --- PLOT 1: The Token Tax (Average Steps Taken) ---
    plt.figure(figsize=(8, 6))
    steps_means = [df['baseline_steps_taken'].mean(), df['anchored_steps_taken'].mean()]
    
    # Using 'hue' and setting 'legend=False' is the modern Seaborn standard to avoid warnings
    ax = sns.barplot(
        x=['Baseline Agent', 'Anchored Agent'], 
        y=steps_means, 
        hue=['Baseline Agent', 'Anchored Agent'], 
        palette=['#ff9999', '#66b3ff'], 
        legend=False
    )
    plt.title('Average Tool Calls per Query (The "Token Tax")', fontsize=14, pad=15)
    plt.ylabel('Average Number of Steps', fontsize=12)
    
    # Add exact numbers on top of the bars
    for i, v in enumerate(steps_means):
        ax.text(i, v + 0.05, str(round(v, 2)), ha='center', fontsize=12, fontweight='bold')
        
    plt.savefig('data/plots/01_token_tax.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 2: The Speed Penalty (Latency) ---
    plt.figure(figsize=(8, 6))
    latency_means = [df['baseline_latency_sec'].mean(), df['anchored_latency_sec'].mean()]
    ax = sns.barplot(
        x=['Baseline Agent', 'Anchored Agent'], 
        y=latency_means, 
        hue=['Baseline Agent', 'Anchored Agent'], 
        palette=['#ffcc99', '#99ff99'], 
        legend=False
    )
    plt.title('Average Response Latency', fontsize=14, pad=15)
    plt.ylabel('Seconds', fontsize=12)

    for i, v in enumerate(latency_means):
        ax.text(i, v + 0.1, f"{round(v, 2)}s", ha='center', fontsize=12, fontweight='bold')

    plt.savefig('data/plots/02_latency_penalty.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- PLOT 3: Distraction Rate ---
    # Calculates how often the agent took more than 1 step (implying it chased a bait tool)
    baseline_distracted = (df['baseline_steps_taken'] > 1).mean() * 100
    anchored_distracted = (df['anchored_steps_taken'] > 1).mean() * 100

    plt.figure(figsize=(8, 6))
    distraction_rates = [baseline_distracted, anchored_distracted]
    ax = sns.barplot(
        x=['Baseline Agent', 'Anchored Agent'], 
        y=distraction_rates, 
        hue=['Baseline Agent', 'Anchored Agent'], 
        palette=['#c2c2f0', '#ffb3e6'], 
        legend=False
    )
    plt.title('Distraction Rate (% of queries using excess tools)', fontsize=14, pad=15)
    plt.ylabel('Percentage (%)', fontsize=12)

    for i, v in enumerate(distraction_rates):
        ax.text(i, v + 1, f"{round(v, 1)}%", ha='center', fontsize=12, fontweight='bold')

    plt.savefig('data/plots/03_distraction_rate.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Success! High-resolution graphs have been saved to the 'data/plots' folder.")

if __name__ == "__main__":
    generate_visualizations()