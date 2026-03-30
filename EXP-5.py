# Step 1: Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

# Step 2: Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data       # Features
y = housing.target     # Target variable

# Step 3: Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Initialize Decision Tree Regressor
dt_reg = DecisionTreeRegressor(criterion='squared_error', random_state=42)

# Step 5: Train the model
dt_reg.fit(X_train, y_train)

# Step 6: Predict values
y_pred = dt_reg.predict(X_test)

# Step 7: Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Predicted Values (first 10):\n", y_pred[:10])
print("\nMean Squared Error:", mse)
print("R2 Score:", r2)