import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error,mean_squared_error, r2_score
from model import build_model
from load_data import load_data

df = load_data("cleaned_data.csv")

X = df[['experience']]
y = df[['salary']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_model()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", root_mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Regression line
plt.scatter(X, y)
plt.plot(X, model.predict(X), linewidth=2)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Linear Regression Fit")
plt.show()

residuals = y_test - y_pred
plt.scatter(X_test, residuals)
plt.axhline(0)
plt.xlabel("Experience")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()