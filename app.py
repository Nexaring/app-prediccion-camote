# =========================================================
# APP DE PREDICCIÓN DE PRODUCCIÓN DE CAMOTE
# STREAMLIT + CATBOOST
# NEXARING
# CON VALORES IDEALES AGRONÓMICOS Y GRÁFICO DINÁMICO
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from catboost import CatBoostRegressor

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
st.set_page_config(
    page_title="NEXARING - Predicción de Camote",
    page_icon="🌱",
    layout="centered"
)

# =========================================================
# CARGAR MODELO Y ARCHIVOS
# =========================================================
@st.cache_resource
def cargar_modelo():
    modelo = CatBoostRegressor()
    modelo.load_model("modelo_camote.cbm")
    return modelo


@st.cache_data
def cargar_json(nombre_archivo):
    with open(nombre_archivo, "r") as f:
        return json.load(f)


modelo = cargar_modelo()
columnas_modelo = cargar_json("columnas_modelo.json")
rangos = cargar_json("rangos_variables.json")
metricas = cargar_json("metricas_modelo.json")

# =========================================================
# VALORES IDEALES AGRONÓMICOS
# =========================================================
valores_ideales = {
    "MesInicioPlantacion": 6,
    "DuracionPlantacion_dias": 90,
    "Temperatura_C": 26,
    "Precipitacion_mm": 564,
    "Humedad_porcentaje": 70,
    "Fertilizante": 80,
    "Riego_Etapa1_mm": 35,
    "Riego_Etapa2_mm": 45,
    "Riego_Etapa3_mm": 55,
    "TipoSuelo": 1,
    "pH_Suelo": 5.5,
    "Altitud_msnm": 250
}

# =========================================================
# RANGOS IDEALES AGRONÓMICOS
# =========================================================
rangos_ideales = {
    "MesInicioPlantacion": [4, 9],
    "DuracionPlantacion_dias": [90, 120],
    "Temperatura_C": [18, 28],
    "Precipitacion_mm": [400, 1000],
    "Humedad_porcentaje": [50, 90],
    "Fertilizante": [70, 90],
    "Riego_Etapa1_mm": [20, 50],
    "Riego_Etapa2_mm": [30, 60],
    "Riego_Etapa3_mm": [40, 70],
    "TipoSuelo": [1, 1],
    "pH_Suelo": [5, 7.5],
    "Altitud_msnm": [0, 500]
}

# =========================================================
# FUNCIONES AUXILIARES
# =========================================================
def es_variable_ph(col):
    nombre = col.lower()
    return nombre == "ph_suelo" or nombre == "phsuelo" or "ph" in nombre


def obtener_columnas_riego(columnas):
    columnas_riego = []
    for col in columnas:
        if "riego" in col.lower():
            columnas_riego.append(col)
    return columnas_riego


def aplicar_regla_riego(entrada):
    """
    Si Riego Etapa 1 o Riego Etapa 2 es 0,
    la producción será 0.
    """

    columnas_riego = obtener_columnas_riego(columnas_modelo)

    if len(columnas_riego) < 2:
        st.error("No se encontraron correctamente las columnas de riego.")
        return False

    riego_1 = columnas_riego[0]
    riego_2 = columnas_riego[1]

    if entrada[riego_1] == 0 or entrada[riego_2] == 0:
        return True

    return False


def aplicar_regla_mes(entrada, prediccion):
    """
    Si MesInicioPlantacion es 1, 2, 11 o 12,
    la producción total no puede pasar de 100 quintales.
    """

    mes = int(entrada.get("MesInicioPlantacion", 0))

    if mes in [1, 2, 11, 12]:
        prediccion = min(prediccion, 100)
        return prediccion, True

    return prediccion, False


def clasificar_produccion(prediccion):
    if prediccion >= 300:
        return "Producción alta estimada", "success"
    elif prediccion >= 200:
        return "Producción media estimada", "warning"
    elif prediccion > 0:
        return "Producción baja estimada", "error"
    else:
        return "Producción nula estimada", "error"


def esta_en_rango_ideal(variable, valor):
    if variable not in rangos_ideales:
        return False

    minimo = rangos_ideales[variable][0]
    maximo = rangos_ideales[variable][1]

    return minimo <= valor <= maximo


def normalizar_valor(variable, valor):
    """
    Convierte cada valor a escala 0-100 para graficar variables con unidades distintas.
    """

    minimo = rangos[variable]["min"]
    maximo = rangos[variable]["max"]

    if maximo == minimo:
        return 0

    valor_normalizado = ((valor - minimo) / (maximo - minimo)) * 100

    if valor_normalizado < 0:
        valor_normalizado = 0

    if valor_normalizado > 100:
        valor_normalizado = 100

    return valor_normalizado


def crear_grafico_dinamico(entrada):
    """
    Crea gráfico dinámico:
    - Línea verde: valores ideales.
    - Barras azules: valores seleccionados dentro del rango ideal.
    - Barras rojas: valores seleccionados fuera del rango ideal.
    """

    variables = []
    valores_usuario_normalizados = []
    valores_ideales_normalizados = []
    colores_barras = []
    textos_hover = []

    for variable in columnas_modelo:
        valor_usuario = entrada[variable]
        valor_ideal = valores_ideales.get(variable, rangos[variable]["mean"])

        usuario_norm = normalizar_valor(variable, valor_usuario)
        ideal_norm = normalizar_valor(variable, valor_ideal)

        variables.append(variable)
        valores_usuario_normalizados.append(usuario_norm)
        valores_ideales_normalizados.append(ideal_norm)

        if esta_en_rango_ideal(variable, valor_usuario):
            colores_barras.append("#2563eb")
            estado = "Dentro del rango ideal"
        else:
            colores_barras.append("#dc2626")
            estado = "Fuera del rango ideal"

        textos_hover.append(
            f"Variable: {variable}<br>"
            f"Valor seleccionado: {valor_usuario}<br>"
            f"Valor ideal: {valor_ideal}<br>"
            f"Estado: {estado}"
        )

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=variables,
            y=valores_usuario_normalizados,
            name="Valores seleccionados",
            marker_color=colores_barras,
            text=[round(v, 1) for v in valores_usuario_normalizados],
            hovertext=textos_hover,
            hoverinfo="text"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=variables,
            y=valores_ideales_normalizados,
            name="Valores ideales",
            mode="lines+markers",
            line=dict(color="#16a34a", width=4),
            marker=dict(size=9, color="#16a34a")
        )
    )

    fig.update_layout(
        title="Comparación dinámica entre valores seleccionados e ideales",
        xaxis_title="Variables del cultivo",
        yaxis_title="Escala normalizada 0 - 100",
        template="plotly_white",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )

    fig.update_xaxes(tickangle=45)

    return fig


# =========================================================
# INICIALIZAR VALORES DE LOS SLIDERS
# =========================================================
for col in columnas_modelo:
    key_slider = f"slider_{col}"

    if key_slider not in st.session_state:
        if col in valores_ideales:
            valor_inicial = valores_ideales[col]
        else:
            valor_inicial = rangos[col]["mean"]

        if es_variable_ph(col):
            st.session_state[key_slider] = float(round(valor_inicial, 1))
        else:
            st.session_state[key_slider] = int(round(valor_inicial))


# =========================================================
# ENCABEZADO
# =========================================================
st.title("🌱 Sistema de Predicción de Producción de Camote - Predicción de Camote")
st.subheader("NEXARING")

st.markdown(
    """
    Esta aplicación estima la **Producción Total de camote** usando un modelo 
    de inteligencia artificial basado en **CatBoost Regressor**.
    """
)

# =========================================================
# MÉTRICAS DEL MODELO
# =========================================================
with st.expander("📊 Ver métricas del modelo"):
    st.write(f"**R² Score:** {metricas['r2']:.4f}")
    st.write(f"**MAE:** {metricas['mae']:.4f}")
    st.write(f"**MSE:** {metricas['mse']:.4f}")
    st.write(f"**RMSE:** {metricas['rmse']:.4f}")

# =========================================================
# BOTONES SUPERIORES
# =========================================================
st.markdown("## Controles rápidos")

col_boton_1, col_boton_2 = st.columns(2)

with col_boton_1:
    boton_valores_ideales = st.button(
        "⭐ Valores ideales",
        use_container_width=True
    )

with col_boton_2:
    boton_reiniciar_promedios = st.button(
        "🔄 Promedios del dataset",
        use_container_width=True
    )

if boton_valores_ideales:
    for col in columnas_modelo:
        key_slider = f"slider_{col}"

        if col in valores_ideales:
            valor = valores_ideales[col]
        else:
            valor = rangos[col]["mean"]

        if es_variable_ph(col):
            st.session_state[key_slider] = float(round(valor, 1))
        else:
            st.session_state[key_slider] = int(round(valor))

    st.success("Se cargaron los valores ideales agronómicos.")
    st.rerun()

if boton_reiniciar_promedios:
    for col in columnas_modelo:
        key_slider = f"slider_{col}"

        valor = rangos[col]["mean"]

        if es_variable_ph(col):
            st.session_state[key_slider] = float(round(valor, 1))
        else:
            st.session_state[key_slider] = int(round(valor))

    st.info("Se cargaron los valores promedio del dataset.")
    st.rerun()

# =========================================================
# FORMULARIO DE ENTRADA
# =========================================================
st.markdown("## Ingrese o ajuste los valores")

entrada_usuario = {}

for col in columnas_modelo:
    valor_min = rangos[col]["min"]
    valor_max = rangos[col]["max"]

    key_slider = f"slider_{col}"

    if es_variable_ph(col):
        entrada_usuario[col] = st.slider(
            label=col,
            min_value=float(valor_min),
            max_value=float(valor_max),
            value=float(st.session_state[key_slider]),
            step=0.1,
            key=key_slider
        )
    else:
        entrada_usuario[col] = st.slider(
            label=col,
            min_value=int(round(valor_min)),
            max_value=int(round(valor_max)),
            value=int(st.session_state[key_slider]),
            step=1,
            key=key_slider
        )

# =========================================================
# TABLA DE VALORES IDEALES
# =========================================================
with st.expander("🌿 Ver valores ideales usados"):
    tabla_ideales = []

    for variable in columnas_modelo:
        ideal = valores_ideales.get(variable, rangos[variable]["mean"])
        rango_ideal = rangos_ideales.get(variable, ["No definido", "No definido"])

        tabla_ideales.append({
            "Variable": variable,
            "Valor ideal": ideal,
            "Rango ideal mínimo": rango_ideal[0],
            "Rango ideal máximo": rango_ideal[1]
        })

    st.dataframe(pd.DataFrame(tabla_ideales), use_container_width=True)

# =========================================================
# GRÁFICO DINÁMICO
# =========================================================
st.markdown("## 📊 Gráfico dinámico de comparación")

fig = crear_grafico_dinamico(entrada_usuario)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Nota: el gráfico usa una escala normalizada de 0 a 100 porque las variables tienen unidades diferentes."
)

# =========================================================
# BOTÓN DE PREDICCIÓN
# =========================================================
st.markdown("---")

boton_predecir = st.button(
    "🔍 Predecir producción",
    use_container_width=True
)

# =========================================================
# PREDICCIÓN
# =========================================================
if boton_predecir:

    entrada_df = pd.DataFrame([entrada_usuario], columns=columnas_modelo)

    if "MesInicioPlantacion" in entrada_usuario:
        st.write("Mes seleccionado:", entrada_usuario["MesInicioPlantacion"])
    else:
        st.error("No se encontró la variable MesInicioPlantacion.")

    # =====================================================
    # REGLA 1: SI FALTA RIEGO EN ETAPA 1 O 2, PRODUCCIÓN = 0
    # =====================================================
    if aplicar_regla_riego(entrada_usuario):
        prediccion = 0.0

        st.error("Producción nula estimada por falta de riego.")
        st.warning("Riego Etapa 1 o Riego Etapa 2 está en 0.")
        st.metric(
            "Predicción de Producción Total",
            f"{prediccion:.2f} quintales"
        )

    else:
        # =================================================
        # PREDICCIÓN DEL MODELO
        # =================================================
        prediccion = modelo.predict(entrada_df)[0]

        if prediccion < 0:
            prediccion = 0

        # =================================================
        # REGLA 2: MESES 1, 2, 11 Y 12 NO PASAN DE 100
        # =================================================
        prediccion, regla_mes_aplicada = aplicar_regla_mes(
            entrada_usuario,
            prediccion
        )

        mensaje, tipo = clasificar_produccion(prediccion)

        if tipo == "success":
            st.success(mensaje)
        elif tipo == "warning":
            st.warning(mensaje)
        else:
            st.error(mensaje)

        if regla_mes_aplicada:
            st.warning(
                "Se aplicó la regla del mes de siembra: "
            )

        st.metric(
            "Predicción de Producción Total",
            f"{prediccion:.2f} quintales"
        )

# =========================================================
# INFORMACIÓN FINAL
# =========================================================
st.markdown("---")
st.caption("Desarrollado para predicción agrícola de producción de camote.")
