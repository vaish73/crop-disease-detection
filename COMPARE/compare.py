import matplotlib
# This line is CRITICAL: It prevents the TclError by not opening a GUI window
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np

# 1. DATA COLLECTION (Based on your project documents)
models = ['ResNet50', 'MobileNetV2', 'DenseNet', 'Simple CNN']
accuracies = [91.13, 82.22, 70.00, 65.00]
params_m = [23.5, 2.26, 7.0, 1.0]  # Parameters in Millions
model_sizes = [98.98, 8.6, 26.87, 4.0]  # Size in MB

# --- GRAPH 1: ACCURACY BAR CHART ---
def generate_accuracy_chart():
    plt.figure(figsize=(10, 6))
    colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']
    
    bars = plt.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.8)
    
    # Adding formatting
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Model Architecture', fontsize=12, fontweight='bold')
    plt.title('Comparison of Model Accuracies for Rice Leaf Disease Detection', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add accuracy labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300)
    print("Successfully saved: accuracy_comparison.png")

# --- GRAPH 2: EFFICIENCY SCATTER PLOT ---
def generate_efficiency_plot():
    plt.figure(figsize=(10, 6))
    
    # We use a scatter plot to show the trade-off
    plt.scatter(params_m, accuracies, s=np.array(model_sizes)*10, color='purple', alpha=0.5, edgecolors='black')
    
    # Annotate each point with the model name
    for i, txt in enumerate(models):
        plt.annotate(txt, (params_m[i], accuracies[i]), textcoords="offset points", 
                     xytext=(0,10), ha='center', fontweight='bold')

    plt.xlabel('Model Complexity (Millions of Parameters)', fontsize=12, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Accuracy vs. Complexity Trade-off', fontsize=14, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Use log scale for X-axis because the difference between 1M and 23M is huge
    plt.xscale('log') 
    
    plt.tight_layout()
    plt.savefig('efficiency_analysis.png', dpi=300)
    print("Successfully saved: efficiency_analysis.png")

# RUN GENERATION
if __name__ == "__main__":
    generate_accuracy_chart()
    generate_efficiency_plot()
    print("\nAll charts generated! You can now find them in your folder.")