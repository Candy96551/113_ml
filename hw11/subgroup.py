import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
import numpy as np

data = load_diabetes()
X = data.data
y = data.target

model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(X, y)

feature_index = 0
X_feature = X[:, feature_index].reshape(-1, 1)

model_single = RandomForestRegressor(n_estimators=100, random_state=0)
model_single.fit(X_feature, y)

x_range = np.linspace(X_feature.min(), X_feature.max(), 100).reshape(-1, 1)
y_pred = model_single.predict(x_range)

plt.scatter(X_feature, y, alpha=0.5, label="True")
plt.plot(x_range, y_pred, color="red", label="Prediction")
plt.xlabel(data.feature_names[feature_index])
plt.ylabel("Disease Progression")
plt.title("Random Forest Regression (Single Feature)")
plt.legend()
plt.show()