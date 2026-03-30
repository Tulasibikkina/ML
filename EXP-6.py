# Step 1: Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load dataset (Iris dataset example)
iris = load_iris()
X = iris.data      # Features
y = iris.target    # Labels

# Step 3: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Initialize Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)

# Step 5: Train the model
rf_model.fit(X_train, y_train)

# Step 6: Predict output
y_pred = rf_model.predict(X_test)

# Step 7: Evaluate performance
print("Predicted Labels:\n", y_pred)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))