# Import required libraries
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

# Step 1: Load dataset
# Sample dataset (2D points)
data = {
    'X': [1, 2, 3, 8, 9, 10],
    'Y': [2, 3, 4, 7, 8, 9]
}

df = pd.DataFrame(data)

# Step 2: Split dataset
X = df[['X', 'Y']]
X_train, X_test = train_test_split(X, test_size=0.25, random_state=0)

# Step 3: Initialize model
model = GaussianMixture(n_components=2, random_state=0)

# Step 4: Train model
model.fit(X_train)

# Step 5: Predict output (cluster labels)
labels = model.predict(X_test)

# Step 6: Evaluate performance
# (Log likelihood is used internally; here we display means)

# Display results
print("Test Data:\n", X_test.values)
print("Predicted Cluster Labels:", labels)
print("Means (Centers):\n", model.means_)