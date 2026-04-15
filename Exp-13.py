# Import required libraries
import numpy as np
import pandas as pd
import skfuzzy as fuzz

# Step 1: Load dataset
# Sample dataset (2D points)
data = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [8, 7],
    [9, 8],
    [10, 9]
])

# Transpose data for skfuzzy (features x samples)
data = data.T

# Step 2: Split dataset
# (Not required for clustering)

# Step 3: Initialize model
n_clusters = 2

# Step 4: Train model
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    data, n_clusters, m=2, error=0.005, maxiter=1000
)

# Step 5: Predict output (cluster membership)
cluster_membership = np.argmax(u, axis=0)

# Step 6: Evaluate performance
# (Use Fuzzy Partition Coefficient - fpc)

# Display results
print("Cluster Centers:\n", cntr)
print("Membership Matrix:\n", u)
print("Final Cluster Labels:", cluster_membership)
print("Fuzzy Partition Coefficient (FPC):", fpc)