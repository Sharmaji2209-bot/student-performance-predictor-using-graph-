import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create dataset
hours = np.array([1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5, 7.5, 8.5])
scores = 25 + (hours * 7) + np.random.randint(-5, 5, size=len(hours))

df = pd.DataFrame({'Hours': hours, 'Score': scores})

# Split data
X = df[['Hours']]
y = df['Score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)
plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Student Performance Prediction")
plt.show()
