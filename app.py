import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Cargar datos
df = pd.read_csv('dataset.csv', sep=';')

# Mapear target
target_map = {'Dropout': 1.0, 'Enrolled': 0.5, 'Graduate': 0.0}
df['Target_Risk'] = df['Target'].map(target_map)

# Features
features = ['Age at enrollment', 'Curricular units 2nd sem (approved)',
            'Curricular units 2nd sem (grade)', 'Gender',
            'Scholarship holder', 'Debtor', 'Tuition fees up to date']

X = df[features]
y = df['Target_Risk']

# Entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# INTERFAZ WEB
st.title("Predicción de Deserción Universitaria")

edad = st.number_input("Edad")
aprobados = st.number_input("Materias aprobadas")
nota = st.number_input("Promedio (0-20)")
genero = st.selectbox("Género", [0,1])
beca = st.selectbox("Beca", [0,1])
deudor = st.selectbox("Deuda", [0,1])
pagos = st.selectbox("Pagos al día", [0,1])

if st.button("Calcular riesgo"):
    datos = pd.DataFrame([[edad, aprobados, nota, genero, beca, deudor, pagos]], columns=features)
    datos_scaled = scaler.transform(datos)

    riesgo = model.predict(datos_scaled)[0]
    riesgo = max(0, min(1, riesgo))

    st.write(f"Probabilidad: {riesgo:.2%}")

    if riesgo > 0.7:
        st.error("Riesgo ALTO")
    elif riesgo > 0.4:
        st.warning("Riesgo MEDIO")
    else:
        st.success("Riesgo BAJO")