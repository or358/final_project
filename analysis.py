import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf

# --- 1. K-means Implementation (Adapted from HW1) ---
def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

def run_kmeans(points, k, max_iter=300, epsilon=1e-4):
    centroids = [p[:] for p in points[:k]]
    d = len(points[0])
    iteration_count = 0
    
    while iteration_count < max_iter:
        clusters = [[] for _ in range(k)]
        
        # Step 1: Assign every point to the closest cluster
        for x in points:
            min_dist = float('inf')
            closest_index = -1
            for i in range(k):
                dist = euclidean_distance(x, centroids[i])
                if dist < min_dist:
                    min_dist = dist
                    closest_index = i
            clusters[closest_index].append(x)
        
        # Step 2 & 3: Update centroids and Check convergence
        new_centroids = []
        converged = True
        
        for i in range(k):
            current_cluster = clusters[i]
            if not current_cluster:
                new_centroids.append(centroids[i])
                continue
            
            cluster_size = len(current_cluster)
            new_mu = [0.0] * d
            for point in current_cluster:
                for dim in range(d):
                    new_mu[dim] += point[dim]
            for dim in range(d):
                new_mu[dim] /= cluster_size
            
            new_centroids.append(new_mu)
            if euclidean_distance(centroids[i], new_mu) >= epsilon:
                converged = False
        
        centroids = new_centroids
        iteration_count += 1
        if converged:
            break
            
    # Final assignment to get the labels for silhouette_score
    labels = [-1] * len(points)
    for pt_idx, x in enumerate(points):
        min_dist = float('inf')
        closest_index = -1
        for i in range(k):
            dist = euclidean_distance(x, centroids[i])
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        labels[pt_idx] = closest_index

    return labels

# --- 2. SymNMF Implementation ---
def init_h(W, n, k):
    m = np.mean(W)
    upper_bound = 2 * np.sqrt(m / k)
    np.random.seed(1234)
    H = np.random.uniform(low=0.0, high=upper_bound, size=(n, k))
    return H.tolist()

def run_symnmf(data, k):
    n = len(data)
    d = len(data[0])
    
    # Run the C extension functions
    W = symnmf.norm(data, n, d)
    H_init = init_h(W, n, k)
    H = symnmf.symnmf(W, H_init, n, k)
    
    # Derive hard clustering (Section 1.5)
    labels = []
    for row in H:
        best_cluster = max(range(len(row)), key=row.__getitem__)
        labels.append(best_cluster)
        
    return labels

# --- 3. Main Analysis Flow ---
def main():
    args = sys.argv
    if len(args) != 3:
        print("An Error Has Occurred")
        sys.exit(1)
        
    try:
        k = int(args[1])
        filename = args[2]
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)
        
    # Read data
    data = []
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.strip():
                    data.append([float(x) for x in line.split(',')])
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)
        
    n = len(data)
    if k >= n or k <= 1:
        print("An Error Has Occurred")
        sys.exit(1)
        
    try:
        # Run both algorithms
        kmeans_labels = run_kmeans(data, k)
        symnmf_labels = run_symnmf(data, k)
        
        # Calculate silhouette scores
        score_kmeans = silhouette_score(data, kmeans_labels)
        score_symnmf = silhouette_score(data, symnmf_labels)
        
        # Output exactly as required
        print(f"nmf: {score_symnmf:.4f}")
        print(f"kmeans: {score_kmeans:.4f}")
        
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()