import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

def compute_dtw(s1, s2):
    """
    Simple implementation of Dynamic Time Warping.
    s1, s2: Input sequences (1D arrays for this demo)
    Returns: cost, path
    
    This is a basic O(N*M) implementation.
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    dtw_matrix[:] = np.inf
    dtw_matrix[0, 0] = 0
    
    # Accumulate Cost
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i-1] - s2[j-1])
            # Take minimum of (match, insert, delete)
            last_min = np.min([dtw_matrix[i-1, j],    # Insertion
                               dtw_matrix[i, j-1],    # Deletion
                               dtw_matrix[i-1, j-1]]) # Match
            dtw_matrix[i, j] = cost + last_min
            
    # Traceback Path
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i-1, j-1))
        if i == 0: j -= 1
        elif j == 0: i -= 1
        else:
            options = [dtw_matrix[i-1, j-1], dtw_matrix[i-1, j], dtw_matrix[i, j-1]]
            best_idx = np.argmin(options)
            if best_idx == 0: 
                i, j = i-1, j-1
            elif best_idx == 1: 
                i -= 1
            else: 
                j -= 1
    path.reverse()
    return dtw_matrix[n, m], path

def run_demo():
    print("Generating demo signals...")
    t = np.linspace(0, 4*np.pi, 100)
    
    # Signal A: Standard Sine Wave
    reference = np.sin(t)
    
    # Signal B: Slower, Shifted Sine Wave ("The Learner")
    # Stretched by 1.2x, Phase shifted
    t_warped = np.linspace(0, 4*np.pi, 120) 
    learner = np.sin(0.9 * t_warped - 0.5) 
    
    print(f"Reference Length: {len(reference)}")
    print(f"Learner Length:   {len(learner)}")
    
    # 1. Compute DTW
    print("Computing DTW alignment...")
    cost, path = compute_dtw(reference, learner)
    print(f"DTW Cost: {cost:.2f}")
    
    # 2. Visualize
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot 1: The Raw Mismatch
        plt.subplot(2, 1, 1)
        plt.title(f"Before Alignment (Raw Mismatch)")
        plt.plot(reference, label='Reference (Teacher)', color='blue', linewidth=2)
        plt.plot(learner, label='Learner (Slower/Shifted)', color='red', linestyle='--', linewidth=2)
        plt.legend()
        plt.grid(True)
        
        # Plot 2: The Alignment Lines
        plt.subplot(2, 1, 2)
        plt.title("DTW Alignment (Elastic Matching)")
        plt.plot(reference, color='blue', label='Reference', linewidth=2)
        plt.plot(learner, color='red', label='Learner', linestyle='--', linewidth=2)
        
        # Draw gray lines connecting matched points
        # Using a subset of lines to avoid clutter
        step = 5
        for (i, j) in path[::step]:
            plt.plot([i, j], [reference[i], learner[j]], color='gray', alpha=0.5, linewidth=0.5)
            
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        print("Saving plot to 'dtw_demo_plot.png'...")
        plt.savefig('dtw_demo_plot.png')
        plt.show()
        
    except Exception as e:
        print(f"Could not plot: {e}")
        print("Note: Install matplotlib to see the graph.")

if __name__ == "__main__":
    run_demo()
