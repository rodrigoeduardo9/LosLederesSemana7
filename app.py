import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Riesgo de Deserción", layout="wide")
st.title("🎓 Predicción de Riesgo de Deserción Universitaria")

# ---------------------------
# CARGA + LIMPIEZA
# ---------------------------
@st.cache_data
def cargar_datos():
    df = pd.read_csv('dataset.csv', sep=';')

    # Mapear target
    target_map = {'Dropout': 1.0, 'Enrolled': 0.5, 'Graduate': 0.0}
    df['Target_Risk'] = df['Target'].map(target_map)

    # Normalizar categorías comunes
    df.replace({
        'Male': 1, 'Female': 0,
        'Yes': 1, 'No': 0
    }, inplace=True)

    # 🔥 Arreglar columna con formato roto
    col_grade = 'Curricular units 2nd sem (grade)'
    df[col_grade] = df[col_grade].astype(str).str.replace('.', '', regex=False)

    # Columnas a usar
    cols = [
        'Age at enrollment',
        'Curricular units 2nd sem (approved)',
        col_grade,
        'Gender',
        'Scholarship holder',
        'Debtor',
        'Tuition fees up to date'
    ]

    # Convertir a numérico
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Quitar filas inválidas
    df = df.dropna(subset=cols + ['Target_Risk'])

    return df, cols

df, features = cargar_datos()

# ---------------------------
# ENTRENAMIENTO (cacheado)
# ---------------------------
@st.cache_resource
def entrenar(df, features):
    X = df[features]
    y = df['Target_Risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Métricas
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return model, scaler, r2, mse

model, scaler, r2, mse = entrenar(df, features)

# ---------------------------
# DASHBOARD
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Métricas del modelo")
    st.metric("R²", f"{r2:.3f}")
    st.metric("MSE", f"{mse:.3f}")

with col2:
    st.subheader("📈 Importancia de variables")
    coefs = model.coef_
    fig, ax = plt.subplots()
    ax.barh(features, coefs)
    ax.set_xlabel("Peso")
    ax.set_title("Coeficientes del modelo")
    st.pyplot(fig)

st.divider()

# ---------------------------
# INPUTS
# ---------------------------
st.subheader("🧾 Ingresar datos del estudiante")

c1, c2, c3 = st.columns(3)

with c1:
    edad = st.number_input("Edad", 0, 100, 18)
    aprobados = st.number_input("Materias aprobadas", 0, 20, 5)

with c2:
    nota = st.number_input("Promedio (0-20)", 0.0, 20.0, 12.0)
    genero = st.selectbox("Género", [0, 1])

with c3:
    beca = st.selectbox("Beca", [0, 1])
    deudor = st.selectbox("Deudor", [0, 1])
    pagos = st.selectbox("Pagos al día", [0, 1])

# ---------------------------
# PREDICCIÓN
# ---------------------------
if st.button("🚀 Calcular riesgo"):
    datos = pd.DataFrame([[
        edad, aprobados, nota, genero, beca, deudor, pagos
    ]], columns=features)

    datos_scaled = scaler.transform(datos)
    riesgo = model.predict(datos_scaled)[0]
    riesgo = max(0, min(1, riesgo))

    st.subheader(f"🎯 Probabilidad de deserción: {riesgo:.2%}")

    if riesgo > 0.7:
        st.error("RIESGO ALTO")
    elif riesgo > 0.4:
        st.warning("RIESGO MEDIO")
    else:
        st.success("RIESGO BAJO")