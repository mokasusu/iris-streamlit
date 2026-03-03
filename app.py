import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris

st.title("🌸 Iris Classifier")

# Load model đã lưu
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

iris = load_iris()

st.sidebar.header("Nhập thông số hoa")

sepal_length = st.sidebar.slider("Sepal length", 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider("Sepal width", 2.0, 4.5, 3.4)
petal_length = st.sidebar.slider("Petal length", 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider("Petal width", 0.1, 2.5, 0.2)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

prediction = model.predict(input_data)

st.subheader("Kết quả dự đoán:")
st.write("Loài hoa:", iris.target_names[prediction][0])