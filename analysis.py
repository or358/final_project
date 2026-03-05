import sys
import math
import numpy as np
from sklearn.metrics import silhouette_score
import symnmf

# --- 1. K-means Implementation (Adapted from HW1) ---
def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))

def assign_clusters(points, centroids, k):
    clusters = [[] for _ in range(k)]
    for x in points:
        min_dist = float('inf')
        closest = -1
        for i in range(k):
            dist = euclidean_distance(x, centroids[i])
            if dist < min_dist:
                min_dist, closest = dist, i
        clusters[closest].append(x)
    return clusters

def update_centroids(clusters, centroids, k, d, eps):
    new_centroids = []
    converged = True
    for i in range(k):
        if not clusters[i]:
            new_centroids.append(centroids[i])
            continue
        size = len(clusters[i])
        new_mu = [sum(pt[dim] for pt in clusters[i]) / size for dim in range(d)]
        new_centroids.append(new_mu)
        if euclidean_distance(centroids[i], new_mu) >= eps:
            converged = False
    return new_centroids, converged

def get_labels(points, centroids, k):
    labels = []
    for x in points:
        min_dist = float('inf')
        closest = -1
        for i in range(k):
            dist = euclidean_distance(x, centroids[i])
            if dist < min_dist:
                min_dist, closest = dist, i
        labels.append(closest)
    return labels

def run_kmeans(points, k, max_iter=300, eps=1e-4):
    centroids = [p[:] for p in points[:k]]
    d = len(points[0])
    for _ in range(max_iter):
        clusters = assign_clusters(points, centroids, k)
        centroids, converged = update_centroids(clusters, centroids, k, d, eps)
        if converged:
            break
    return get_labels(points, centroids, k)

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

def read_data(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                data.append([float(x) for x in line.split(',')])
    return data

# --- 3. Main Analysis Flow ---
def main():
    args = sys.argv
    if len(args) != 3:
        print("An Error Has Occurred")
        sys.exit(1)
        
    try:
        k = int(args[1])
        filename = args[2]
        data = read_data(filename)
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)
        
    if k >= len(data) or k <= 1:
        print("An Error Has Occurred")
        sys.exit(1)
        
    try:
        score_kmeans = silhouette_score(data, run_kmeans(data, k))
        score_symnmf = silhouette_score(data, run_symnmf(data, k))
        
        print(f"nmf: {score_symnmf:.4f}")
        print(f"kmeans: {score_kmeans:.4f}")
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()