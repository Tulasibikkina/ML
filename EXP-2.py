#EXp2
# Step 1: Import required libraries
import pandas as pd
import numpy as np

# Step 2: Create dataset (or use pd.read_csv)
data = {
    'Age': [25, 30, np.nan, 35, 120, 40, 28],
    'Salary': [50000, 60000, 55000, np.nan, 1000000, 65000, 62000],
    'Experience': [1, 3, 2, 5, 30, np.nan, 4]
}

df = pd.DataFrame(data)

print("Original Dataset:\n", df)

# Step 3: Identify missing values
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Handle missing values (fixed)
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Experience'] = df['Experience'].fillna(df['Experience'].mean())

print("\nAfter Handling Missing Values:\n", df)

# Step 5: Discretization
df['Age_Group'] = pd.cut(df['Age'],
                        bins=[0, 30, 50, 1000],
                        labels=['Young', 'Adult', 'Senior'])

print("\nAfter Discretization:\n", df)

# Step 6: Outlier detection and removal (fixed)
numeric_df = df.select_dtypes(include=[np.number])

Q1 = numeric_df.quantile(0.25)
Q3 = numeric_df.quantile(0.75)
IQR = Q3 - Q1

df_clean = df[~((numeric_df < (Q1 - 1.5 * IQR)) |
                (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)]

print("\nAfter Removing Outliers:\n", df_clean)

# Step 7: Feature selection
selected_features = df_clean[['Age', 'Salary']]

print("\nSelected Features:\n", selected_features)