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

    # Target
    target_map = {'Dropout': 1.0, 'Enrolled': 0.5, 'Graduate': 0.0}
    df['Target_Risk'] = df['Target'].map(target_map)

    # Convertir texto a números
    df.replace({
        'Male': 1, 'Female': 0,
        'Yes': 1, 'No': 0
    }, inplace=True)

    # Arreglar columna problemática
    col_grade = 'Curricular units 2nd sem (grade)'
    df[col_grade] = df[col_grade].astype(str).str.replace('.', '', regex=False)

    # Columnas
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

    # Eliminar filas malas
    df = df.dropna(subset=cols + ['Target_Risk'])

    return df, cols

df, features = cargar_datos()

# ---------------------------
# ENTRENAMIENTO
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

    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    return model, scaler, r2, mse

model, scaler, r2, mse = entrenar(df, features)

# ---------------------------
# MÉTRICAS
# ---------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Métricas del modelo")
    st.metric("R²", f"{r2:.3f}")
    st.metric("MSE", f"{mse:.3f}")

with col2:
    st.subheader("📈 Importancia de variables")
    fig, ax = plt.subplots()
    ax.barh(features, model.coef_)
    ax.set_title("Coeficientes")
    st.pyplot(fig)

st.divider()

# ---------------------------
# INPUTS (PRO)
# ---------------------------
st.subheader("🧾 Datos del estudiante")

c1, c2, c3 = st.columns(3)

with c1:
    edad = st.number_input("Edad", 0, 100, 18)
    aprobados = st.number_input("Materias aprobadas", 0, 20, 5)

with c2:
    nota = st.number_input("Promedio (0-20)", 0.0, 20.0, 12.0)
    genero = st.selectbox("Género", ["Femenino", "Masculino"])

with c3:
    beca = st.selectbox("¿Tiene beca?", ["No", "Sí"])
    deudor = st.selectbox("¿Es deudor?", ["No", "Sí"])
    pagos = st.selectbox("¿Pagos al día?", ["No", "Sí"])

# ---------------------------
# PREDICCIÓN
# ---------------------------
if st.button("🚀 Calcular riesgo"):

    # Convertir a números
    genero_num = 1 if genero == "Masculino" else 0

    mapa = {"Sí": 1, "No": 0}
    beca_num = mapa[beca]
    deudor_num = mapa[deudor]
    pagos_num = mapa[pagos]

    datos = pd.DataFrame([[
        edad, aprobados, nota,
        genero_num, beca_num, deudor_num, pagos_num
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