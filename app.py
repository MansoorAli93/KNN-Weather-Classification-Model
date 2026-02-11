import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
st.set_page_config(page_title="KNN Weather Classifier", layout="centered")
st.title("KNN Weather Classification")
st.write("Predict whether the weather will be **Sunny** or **Rainy** using KNN.")
X = np.array([
    [50, 70],
    [25, 80],
    [27, 60],
    [31, 65],
    [23, 85],
    [20, 75]
])
y = np.array([0, 1, 0, 0, 1, 1])
label_map = {0: "Sunny", 1: "Rainy"}
st.sidebar.header("Input Features")
temp = st.sidebar.slider("Temperature (°C)", 10, 60, 26)
hum = st.sidebar.slider("Humidity (%)", 50, 95, 78)
k_value = st.sidebar.slider("K Value (Neighbors)", 1, 5, 3)
knn = KNeighborsClassifier(n_neighbors=k_value)
knn.fit(X, y)
new_data = np.array([[temp, hum]])
prediction = knn.predict(new_data)[0]
if prediction == 1:
    st.error(f"Predicted Weather: **{label_map[prediction]}** 🌧️")
else:
    st.success(f"Predicted Weather: **{label_map[prediction]}** ☀️")
if st.checkbox("Show Training Data"):
    st.write("Training Dataset:")
    st.dataframe({
        "Temperature": X[:, 0],
        "Humidity": X[:, 1],
        "Label": [label_map[val] for val in y]
    })
fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], color='orange', label='Sunny', s=100, edgecolor='k')
ax.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', label='Rainy', s=100, edgecolor='k')
ax.scatter(temp, hum, color='red' if prediction == 1 else 'orange', marker='*', s=300, edgecolor='black', label=f'New Day: {label_map[prediction]}')
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Humidity (%)')
ax.set_title('KNN Weather Classification')
ax.legend()
ax.grid(True)
st.pyplot(fig)
st.caption("Built with Streamlit & Scikit-learn")


