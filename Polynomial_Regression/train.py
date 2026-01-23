import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("Polynomial_Regression/data.csv")

X = data[['hours_studied']]
y = data['exam_score']

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_test_point = pd.DataFrame([[6]], columns=['hours_studied'])
prediction = model.predict(poly.transform(X_test_point))

print(f"Predicted score for 6 hours: {prediction[0]:.2f}")

y_train_pred = model.predict(X_poly)
r2 = r2_score(y, y_train_pred)
print(f"R2_score =", r2)
X_range = np.linspace(X.values.min(), X.values.max(), 100).reshape(-1, 1)
X_range_df = pd.DataFrame(X_range, columns=['hours_studied'])

y_pred_curve = model.predict(poly.transform(X_range_df))

plt.figure()
plt.scatter(X, y)
plt.plot(X_range, y_pred_curve)
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title(f"Polynomial Regression (Degree 2) | R² = {r2:.4f}")
plt.show()
