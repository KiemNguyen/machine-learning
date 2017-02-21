# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
polynomial_regressor = PolynomialFeatures(degree = 4)
X_polynomial = polynomial_regressor.fit_transform(X)

# Create a new linear regressor object
linear_regressor2 = LinearRegression()
linear_regressor2.fit(X_polynomial, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
# Plot the prediction line
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')

# Visualising the Polynomial Regression results
X_grid = np.arange(min(X), max(X), 0.1)
# Reshape to create a matrix from a vector
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(X, y, color = 'red')
# Plot the prediction
plt.plot(X_grid, linear_regressor2.predict(polynomial_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Potision level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
linear_regressor.predict(6.5)

# Predicting a new result with Polynomial Regression
linear_regressor2.predict(polynomial_regressor.fit_transform(6.5))