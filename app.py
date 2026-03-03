import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Tiêu đề
st.title("🌸 Iris Flower Classifier")

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Sidebar nhập dữ liệu
st.sidebar.header("Nhập thông số hoa")

sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 0.2)

# Tạo input
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Dự đoán
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Hiển thị kết quả
st.subheader("Kết quả dự đoán:")
st.write("Loài hoa:", iris.target_names[prediction][0])

st.subheader("Xác suất dự đoán:")
st.write(prediction_proba)

# Hiển thị dataset
if st.checkbox("Hiển thị dữ liệu Iris"):
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["species"] = y
    st.write(df)