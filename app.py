import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Membaca data
st.title("Aplikasi Pembaca Data dan Visualisasi Machine Learning")
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    # Membaca file CSV
    data = pd.read_csv(uploaded_file)
    st.write("Data yang diunggah:")
    st.write(data.head())

    # Membagi data
    st.write("Memilih Fitur dan Target")
    features = st.multiselect("Pilih fitur:", data.columns)
    target = st.selectbox("Pilih target:", data.columns)

    if len(features) > 0 and target:
        X = data[features]
        y = data[target]

        # Membagi data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Membuat model regresi linier
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prediksi dan evaluasi
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        st.write("MSE (Mean Squared Error):", mse)

        # Menampilkan visualisasi
        if st.button("Tampilkan Plot Hasil"):
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.7)
            plt.xlabel("Nilai Sebenarnya")
            plt.ylabel("Nilai Prediksi")
            plt.title("Visualisasi Hasil Prediksi vs Nilai Sebenarnya")
            st.pyplot(plt)

