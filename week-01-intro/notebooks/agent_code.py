import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv(r"c:\Users\princ\Desktop\MachineLearning2026\MachineLearning2026_PrinceSachan\ml_course\Dataset\medical.csv")

# Filter for non-smokers only
df = df[df["smoker"] == "no"]

# Features and target
X = df[["bmi"]].values
y = df["charges"].values

# Train model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color="steelblue", alpha=0.5, label="Data Points")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regression Line")
plt.xlabel("BMI")
plt.ylabel("Charges")
plt.title("Linear Regression: BMI vs Medical Charges")
plt.legend()
plt.tight_layout()
plt.show()
