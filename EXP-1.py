import numpy as np
import statistics
# Dataset
data = [1,2,3,4,5,6,7,8,9,10]
# Mean
mean_res = np.mean(data)
# Median
median_res = np.median(data)
# Mode
mode_res = statistics.mode(data)
# Standard Deviation
std_dev = np.std(data)
# Variance
variance = np.var(data)
# Display results
print(f"Mean: {mean_res}")
print(f"Median: {median_res}")
print(f"Mode: {mode_res}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")