# Import required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load dataset
# Sample dataset (Student pass/fail based on study hours)
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Pass':  [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

# Step 2: Split dataset
X = df[['Hours']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Step 3: Initialize model
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=0)

# Step 4: Train model
model.fit(X_train, y_train)

# Step 5: Predict output
y_pred = model.predict(X_test)

# Step 6: Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display results
print("Predicted Output:", y_pred)
print("Actual Output:", y_test.values)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)