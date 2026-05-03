import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.title("Predicción de Riesgo de Deserción Universitaria")

# ---------------------------
# CARGA DE DATOS
# ---------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv('dataset.csv', sep=';')

    # Mapear variable target
    target_map = {'Dropout': 1.0, 'Enrolled': 0.5, 'Graduate': 0.0}
    df['Target_Risk'] = df['Target'].map(target_map)

    # Convertir columnas a numérico (evita errores)
    columnas = ['Gender', 'Scholarship holder', 'Debtor', 'Tuition fees up to date']
    for col in columnas:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Eliminar nulos
    df = df.fillna(0)

    return df

df = cargar_datos()

# ---------------------------
# FEATURES
# ---------------------------
features = [
    'Age at enrollment',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Gender',
    'Scholarship holder',
    'Debtor',
    'Tuition fees up to date'
]

X = df[features]
y = df['Target_Risk']

# ---------------------------
# ENTRENAMIENTO
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------
# INTERFAZ
# ---------------------------
st.header("Ingresa los datos del estudiante")

edad = st.number_input("Edad", min_value=0.0)
aprobados = st.number_input("Materias aprobadas", min_value=0.0)
nota = st.number_input("Promedio (0-20)", min_value=0.0, max_value=20.0)

genero = st.selectbox("Género", [0, 1])
beca = st.selectbox("¿Tiene beca?", [0, 1])
deudor = st.selectbox("¿Es deudor?", [0, 1])
pagos = st.selectbox("¿Pagos al día?", [0, 1])

# ---------------------------
# PREDICCIÓN
# ---------------------------
if st.button("Calcular riesgo"):
    datos = pd.DataFrame([[edad, aprobados, nota, genero, beca, deudor, pagos]], columns=features)

    datos_scaled = scaler.transform(datos)

    riesgo = model.predict(datos_scaled)[0]
    riesgo = max(0, min(1, riesgo))

    st.subheader(f"Probabilidad de deserción: {riesgo:.2%}")

    if riesgo > 0.7:
        st.error("RIESGO ALTO")
    elif riesgo > 0.4:
        st.warning("RIESGO MEDIO")
    else:
        st.success("RIESGO BAJO")