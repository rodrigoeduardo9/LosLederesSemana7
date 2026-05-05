import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Dashboard Deserción", layout="wide")

# ---------------------------
# 🎨 ESTILO CLARO PROFESIONAL
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
    color: #1f2937;
}
.block-container {
    padding: 2rem;
}
h1 {
    color: #4f46e5;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 8px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
section[data-testid="stSidebar"] {
    background-color: #ffffff;
}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Dashboard de Riesgo de Deserción")
st.caption("Análisis predictivo basado en Machine Learning")

# ---------------------------
# 📊 CARGA Y LIMPIEZA
# ---------------------------
@st.cache_data
def cargar():
    df = pd.read_csv('dataset.csv', sep=';')

    df.replace({
        'Male': 1, 'Female': 0,
        'Yes': 1, 'No': 0
    }, inplace=True)

    df['Curricular units 2nd sem (grade)'] = df['Curricular units 2nd sem (grade)']\
        .astype(str).str.replace('.', '', regex=False)

    cols = [
        'Age at enrollment',
        'Curricular units 2nd sem (approved)',
        'Curricular units 2nd sem (grade)',
        'Gender',
        'Scholarship holder',
        'Debtor',
        'Tuition fees up to date'
    ]

    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['Target_Risk'] = df['Target'].map({
        'Dropout': 1, 'Enrolled': 0.5, 'Graduate': 0
    })

    df = df.dropna()

    return df, cols

df, features = cargar()

# ---------------------------
# 🤖 MODELO
# ---------------------------
@st.cache_resource
def entrenar():
    X = df[features]
    y = df['Target_Risk']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, scaler, r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)

model, scaler, r2, mse = entrenar()

# ---------------------------
# 📊 MÉTRICAS
# ---------------------------
c1, c2 = st.columns(2)

with c1:
    st.markdown(f"""
    <div class="card">
        <h3>📈 R² del modelo</h3>
        <h1>{r2:.3f}</h1>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <h3>📉 Error (MSE)</h3>
        <h1>{mse:.3f}</h1>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# 📈 GRÁFICO
# ---------------------------
st.subheader("Factores que influyen en la deserción")

fig = px.bar(
    x=model.feature_importances_,
    y=features,
    orientation='h',
    color=model.feature_importances_,
    color_continuous_scale='Blues'
)

fig.update_layout(
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_color='#1f2937'
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# 🧾 INPUTS
# ---------------------------
st.subheader("Evaluar estudiante")

c1, c2, c3 = st.columns(3)

with c1:
    edad = st.number_input("Edad", 0, 100, 18)
    aprobados = st.number_input("Cursos aprobados", 0, 20, 5)

with c2:
    nota = st.number_input("Promedio", 0.0, 20.0, 12.0)
    genero = st.selectbox("Género", ["Femenino", "Masculino"])

with c3:
    beca = st.selectbox("Beca", ["No", "Sí"])
    deudor = st.selectbox("Deuda", ["No", "Sí"])
    pagos = st.selectbox("Pagos al día", ["No", "Sí"])

# ---------------------------
# 🔮 PREDICCIÓN
# ---------------------------
if st.button("Calcular riesgo"):

    genero = 1 if genero == "Masculino" else 0
    mapa = {"Sí": 1, "No": 0}

    datos = pd.DataFrame([[
        edad, aprobados, nota,
        genero, mapa[beca], mapa[deudor], mapa[pagos]
    ]], columns=features)

    datos = scaler.transform(datos)

    riesgo = model.predict(datos)[0]
    riesgo = max(0, min(1, riesgo))

    st.markdown(f"""
    <div class="card">
        <h2>Resultado</h2>
        <h1>{riesgo:.2%}</h1>
    </div>
    """, unsafe_allow_html=True)

    st.progress(riesgo)

    if riesgo > 0.7:
        st.error("Riesgo alto")
    elif riesgo > 0.4:
        st.warning("Riesgo medio")
    else:
        st.success("Riesgo bajo")