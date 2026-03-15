import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('Salary_Data.csv')

# Extract variables
x = data['YearsExperience'].values
y = data['Salary'].values

# Means
mean_x = np.mean(x)
mean_y = np.mean(y)

# Least squares slope and intercept
numerator = 0
denominator = 0
for i in range(len(x)):
    numerator += (x[i] - mean_x) * (y[i] - mean_y)
    denominator += (x[i] - mean_x) ** 2

m = numerator / denominator   # slope
c = mean_y - (mean_x * m)     # intercept

# Generate regression line values
x_vals = np.linspace(np.min(x), np.max(x), 50)
y_vals = m * x_vals + c

# Plot
plt.scatter(x, y, color='#FF0000', label='Data points')
plt.plot(x_vals, y_vals, color='#0000FF', label='Regression line')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()