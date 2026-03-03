import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dữ liệu
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Lưu model thành file .pkl
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model đã được lưu thành iris_model.pkl")