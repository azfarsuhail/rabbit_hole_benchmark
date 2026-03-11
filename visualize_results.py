import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def generate_research_plots():
    # Load Data
    try:
        df = pd.read_csv("data/benchmark_results.csv")
    except FileNotFoundError:
        print("Error: benchmark_results.csv not found in data/ folder.")
        return

    # Create output directory
    os.makedirs("data/plots", exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    
    # ---------------------------------------------------------
    # PLOT 1: Cumulative Token Tax (Line Graph)
    # PROVES: Long-term cost scalability.
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(df['task_id'], df['baseline_steps_taken'].cumsum(), label='Baseline (Unanchored)', color='#e74c3c', lw=2)
    plt.plot(df['task_id'], df['anchored_steps_taken'].cumsum(), label='Anchored (Veritas)', color='#3498db', lw=2)
    plt.fill_between(df['task_id'], df['anchored_steps_taken'].cumsum(), df['baseline_steps_taken'].cumsum(), color='gray', alpha=0.15, label='Wasted Compute Area')
    plt.title('Figure 1: Cumulative Computational Divergence', fontsize=14)
    plt.xlabel('Query Sequence')
    plt.ylabel('Total Cumulative Tool Calls')
    plt.legend()
    plt.savefig('data/plots/01_cumulative_tax.png', dpi=300)

    # ---------------------------------------------------------
    # PLOT 2: Step Distribution (Histogram)
    # PROVES: Reliability. High steps = "Rabbit Hole" failures.
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plot_data = pd.melt(df[['baseline_steps_taken', 'anchored_steps_taken']], var_name='Agent', value_name='Steps')
    sns.countplot(data=plot_data, x='Steps', hue='Agent', palette='Set2')
    plt.title('Figure 2: Reliability Distribution (Tool Call Frequency)', fontsize=14)
    plt.xlabel('Number of Steps (Tool Calls)')
    plt.ylabel('Frequency (Count)')
    plt.savefig('data/plots/02_step_distribution.png', dpi=300)

    # ---------------------------------------------------------
    # PLOT 3: Latency Density (KDE Plot)
    # PROVES: Speed consistency.
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df['baseline_latency_sec'], label='Baseline', fill=True, color='#e74c3c')
    sns.kdeplot(df['anchored_latency_sec'], label='Anchored', fill=True, color='#3498db')
    plt.title('Figure 3: Response Latency Probability Density', fontsize=14)
    plt.xlabel('Latency (Seconds)')
    plt.ylabel('Density')
    plt.savefig('data/plots/03_latency_density.png', dpi=300)

    # ---------------------------------------------------------
    # PLOT 4: Box Plot of Latency (Outlier Detection)
    # PROVES: Anchoring prevents "Spiral" events.
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=plot_data, x='Agent', y='Steps', palette='Pastel1')
    plt.title('Figure 4: Variance and Outlier Analysis', fontsize=14)
    plt.savefig('data/plots/04_step_variance.png', dpi=300)

    # ---------------------------------------------------------
    # PLOT 5: Distraction Rate % (Bar Chart)
    # PROVES: Direct success metric.
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    distraction = [
        (df['baseline_steps_taken'] > 1).mean() * 100,
        (df['anchored_steps_taken'] > 1).mean() * 100
    ]
    sns.barplot(x=['Baseline', 'Anchored'], y=distraction, palette='magma')
    plt.title('Figure 5: Total Distraction Rate (%)', fontsize=14)
    plt.ylabel('Percentage of Queries with Excess Tool Usage')
    plt.savefig('data/plots/05_distraction_rate.png', dpi=300)

    # ---------------------------------------------------------
    # PLOT 6: Efficiency Ratio (Scatter Plot)
    # PROVES: Latency vs. Steps correlation.
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.scatter(df['baseline_steps_taken'], df['baseline_latency_sec'], alpha=0.3, label='Baseline', color='#e74c3c')
    plt.scatter(df['anchored_steps_taken'], df['anchored_latency_sec'], alpha=0.3, label='Anchored', color='#3498db')
    plt.title('Figure 6: Step-to-Latency Correlation', fontsize=14)
    plt.xlabel('Steps Taken')
    plt.ylabel('Latency (sec)')
    plt.legend()
    plt.savefig('data/plots/06_efficiency_scatter.png', dpi=300)

    # ---------------------------------------------------------
    # PLOT 7: Rolling Average Latency
    # PROVES: System stability over long-duration runs.
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    df['baseline_rolling'] = df['baseline_latency_sec'].rolling(window=20).mean()
    df['anchored_rolling'] = df['anchored_latency_sec'].rolling(window=20).mean()
    plt.plot(df['task_id'], df['baseline_rolling'], label='Baseline (20-query Moving Avg)', color='#e74c3c')
    plt.plot(df['task_id'], df['anchored_rolling'], label='Anchored (20-query Moving Avg)', color='#3498db')
    plt.title('Figure 7: Temporal Latency Stability', fontsize=14)
    plt.ylabel('Seconds')
    plt.legend()
    plt.savefig('data/plots/07_temporal_stability.png', dpi=300)

    # ---------------------------------------------------------
    # PLOT 8: Error Delta (Heatmap Style)
    # PROVES: Where specifically the Anchored agent saved resources.
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 4))
    delta = (df['baseline_steps_taken'] - df['anchored_steps_taken']).values.reshape(10, 50)
    sns.heatmap(delta, cmap='RdYlGn', center=0, cbar_kws={'label': 'Steps Saved'})
    plt.title('Figure 8: Step-Saving Heatmap (Spatial Delta Analysis)', fontsize=14)
    plt.xlabel('Query Batch')
    plt.savefig('data/plots/08_savings_heatmap.png', dpi=300)

    print("\n[SUCCESS] 8 Research-grade plots generated in data/plots/")

if __name__ == "__main__":
    generate_research_plots()