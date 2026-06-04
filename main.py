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

import pandas as pd


study_data = {
    'Study_Hours': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,7.5,8.5,9.5],
    'Exam_Score': [30, 40, 46, 50, 58, 65, 72, 80, 85, 92,]
}

df = pd.DataFrame(study_data)

print("Full Dataset:")
print(df)

print("\nPreview (Top Rows):")
print(df.head(3))

print("\nSummary Statistics:")
summary = df.describe()
print(summary)

print("\nCheck for Missing Values:")
missing = df.isnull().sum()
print(missing)

import pandas as pd

study_data = {
    'Study_Hours': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    'Exam_Score': [30, 40, 46, 50, 58, 65, 72, 80, 85, 92]
}

df = pd.DataFrame(study_data)

print(df.head())

import pandas as pd

data = {
    'study_hours': [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5],
    'exam_score': [30, 40, 46, 50, 58, 65, 72, 80, 85, 92]
}

df = pd.DataFrame(data)

print(df)


print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

CtgX = df[['study_hours']]
y = df['exam_score']


import pandas as pd
import numpy as np

hours = np.array([1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5, 7.5, 8.5])
scores = 25 + (hours * 7) + np.random.randint(-5, 5, size=len(hours))

df = pd.DataFrame({
    'Hours': hours,
    'Score': scores
})

print(df.head())

from sklearn.model_selection import train_test_split

X = df[['Hours']]
y = df['Score']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X = df[['Hours']]
y = df['Score']

from sklearn.model_selection import train_test_split

model = LinearRegression()

model.fit(X_train, y_train)

print("Model trained successfully ✅")

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

pred = model.predict([[5]])
print("Predicted Score:", pred[0])

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)

if r2 > 0.9:
    print("Excellent model ")
elif r2 > 0.7:
    print("Good model ")
else:
    print("Needs improvement ")

import matplotlib.pyplot as plt

plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.plot(X_test, y_pred, color='red', label="Predicted Line")

plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Actual vs Predicted")
plt.legend()

plt.show()

from sklearn.metrics import mean_absolute_error

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} points")

import pandas as pd

results = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})

print(results)

new_hours = [[4.5]]
predicted_score = model.predict(new_hours)

print(f"\nPredicted score for 4.5 hours: {predicted_score[0]:.2f}")
