import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = {
    'X': [1, 2, 3, 8, 9, 10],
    'Y': [2, 3, 4, 7, 8, 9]
}
df = pd.DataFrame(data)
X = df[['X', 'Y']]
model = KMeans(n_clusters=2, random_state=0)
model.fit(X)
labels = model.predict(X)
print("Data Points:\n", X.values)
print("Cluster Labels:", labels)
print("Centroids:\n", model.cluster_centers_)
plt.scatter(X['X'], X['Y'], c=labels)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], marker='X')
plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()